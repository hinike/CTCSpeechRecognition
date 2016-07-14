require 'UtilsMultiGPU'
require 'BNDecorator'

-- Wraps rnn module into bi-directional.
local function BRNN(model, rnnType, nIn, nHidden, is_cudnn)
    if is_cudnn then
        require 'cudnn'
        local rnn = cudnn.RNN(nIn, nHidden, 1)
        rnn.mode = 'CUDNN_'..rnnType
        if nIn == nHidden then
            rnn.inputMode = 'CUDNN_SKIP_INPUT'
        end
        rnn.bidirectional = 'CUDNN_BIDIRECTIONAL'
        rnn.numDirections = 2
        rnn:reset()
        model:add(rnn)
    else
        require 'rnn'
        local fwdLstm = nn.SeqLSTM(nIn, nHidden)
        local bwdLstm = nn.SeqLSTM(nIn, nHidden)
        local ct = nn.ConcatTable():add(fwdLstm):add(bwdLstm)
        model:add(ct):add(nn.JoinTable(3))
    end
    model:add(nn.View(-1, 2, nHidden):setNumInputDims(2))
    model:add(nn.Sum(3))
    model:add(nn.BNDecorator(nHidden))
end


-- Based on convolution kernel and strides.
local function calculateInputSizes(sizes)
    sizes = torch.floor((sizes - 11) / 2 + 1) -- conv1
    sizes = torch.floor((sizes - 11) + 1) -- conv2
    sizes = torch.floor((sizes - 11) + 1) -- conv3
    return sizes
end


local function get_min_width()
    local width = 1
    width = (width+1) * 2 + 11
    width = (width+1) + 11
    width = (width+1) + 11
    return width
end


local function deepSpeech(rnnType, rnnHiddenSize, nbOfHiddenLayers, dict_size, nGPU, isCUDNN)
    --[[
        Creates the covnet+rnn structure.
        input:
            height: specify the dataHeight, typically 129 for spect; 26 for logfbank
            dict_size = size of dictionary
    --]]
    local model = nn.Sequential()

    -- (nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]) conv layers.
    model:add(nn.SpatialConvolution(1, 32, 11, 41, 2, 2))
    model:add(nn.SpatialBatchNormalization(32, 1e-3))
    model:add(nn.ReLU(true))
    model:add(nn.SpatialConvolution(32, 32, 11, 21, 1, 2))
    model:add(nn.SpatialBatchNormalization(32, 1e-3))
    model:add(nn.ReLU(true))
    -- TODO the DS2 architecture does not include this layer, but mem overhead increases.
    -- model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    model:add(nn.SpatialConvolution(32, 96, 11, 21, 1, 2))
    model:add(nn.SpatialBatchNormalization(96, 1e-3))
    model:add(nn.ReLU(true))

    local rnnInputsize = 96 * 1 -- outputPlanes X outputHeight
    local rnnOutputSize = rnnHiddenSize -- size of rnn output

    model:add(nn.View(rnnInputsize, -1):setNumInputDims(3)) -- batch x models x seqLength
    model:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- seqLength x batch x models

    BRNN(model, rnnType, rnnInputsize, rnnHiddenSize, isCUDNN)

    for i = 1, nbOfHiddenLayers-1 do
        BRNN(model, rnnType, rnnOutputSize, rnnHiddenSize, isCUDNN)
    end

    model:add(nn.View(-1, rnnOutputSize)) -- (seqLength x batch) x models
    model:add(nn.Linear(rnnOutputSize, dict_size))
    model = makeDataParallel(model, nGPU, isCUDNN)
    print (model)
    return model
end


return { deepSpeech, calculateInputSizes, get_min_width }
