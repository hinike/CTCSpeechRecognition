require 'UtilsMultiGPU'
require 'BNDecorator'

-- Wraps rnn module into bi-directional.
local function BLSTM(model, nIn, nHidden, is_cudnn)
    if is_cudnn then
        require 'cudnn'
        model:add(cudnn.BLSTM(nIn, nHidden, 1))
    else
        require 'rnn'
        local fwdLstm = nn.SeqLSTM(nIn, nHidden)
        local bwdLstm = nn.SeqLSTM(nIn, nHidden)
        local ct = nn.ConcatTable():add(fwdLstm):add(bwdLstm)
        model:add(ct):add(nn.JoinTable(3))
    end
    model:add(nn.BNDecorator(cudnn.BatchNormalization(2*nHidden, 1e-3)))
end


-- Based on convolution kernel and strides.
local function calculateInputSizes(sizes)
    sizes = torch.floor((sizes - 11) / 2 + 1) -- conv1
    sizes = torch.floor((sizes - 11) / 2 + 1) -- conv2
    sizes = torch.floor((sizes - 2) / 2 + 1) -- pool1
    return sizes
end


local function get_min_width()
    local width = 1
    width = (width+1) * 2 + 2
    width = (width+1) * 2 + 11
    width = (width+1) * 2 + 11
    return width
end


local function deepSpeech(nGPU, isCUDNN, height, dict_size)
    --[[
        Creates the covnet+rnn structure.
        input:
            height: specify the dataHeight, typically 129 for spect; 26 for logfbank
            dict_size = size of dictionary
    --]]

    local GRU = false
    local model = nn.Sequential()

    -- (nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]) conv layers.
    model:add(nn.SpatialConvolution(1, 32, 11, 41, 2, 2))
    model:add(nn.SpatialBatchNormalization(32, 1e-3))
    model:add(nn.ReLU(true))
    model:add(nn.SpatialConvolution(32, 32, 11, 21, 2, 1))
    model:add(nn.SpatialBatchNormalization(32, 1e-3))
    model:add(nn.ReLU(true))
    -- TODO the DS2 architecture does not include this layer, but mem overhead increases.
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    local rnnInputsize = 32 * 12 -- outputPlanes X outputHeight
    local rnnHiddenSize = 400 -- size of rnn hidden layers
    local rnnOutputSize = 2*rnnHiddenSize -- size of rnn output
    local nbOfHiddenLayers = 4

    model:add(nn.View(rnnInputsize, -1):setNumInputDims(3)) -- batch x models x seqLength
    model:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- seqLength x batch x models

    BLSTM(model, rnnInputsize, rnnHiddenSize, isCUDNN)

    for i = 1, nbOfHiddenLayers do
        BLSTM(model, rnnOutputSize, rnnHiddenSize, isCUDNN)
    end

    model:add(nn.View(-1, rnnOutputSize)) -- (seqLength x batch) x models
    model:add(nn.Linear(rnnOutputSize, dict_size))
    model = makeDataParallel(model, nGPU, isCUDNN)
    return model
end


return { deepSpeech, calculateInputSizes, get_min_width }
