require 'nngraph'
require 'MaskRNN'
require 'ReverseMaskRNN'
require 'UtilsMultiGPU'

-- Chooses RNN based on if GRU or backend GPU support.
local function getRNNModule(nIn, nHidden, GRU, is_cudnn)
    if (GRU) then
        if is_cudnn then
            require 'cudnn'
            return cudnn.GRU(nIn, nHidden, 1)
        else
            require 'rnn'
        end
        return nn.GRU(nIn, nHidden)
    end
    if is_cudnn then
        require 'cudnn'
        return cudnn.LSTM(nIn, nHidden, 1)
    else
        require 'rnn'
    end
    return nn.SeqLSTM(nIn, nHidden)
end


-- Wraps rnn module into bi-directional.
local function BRNN(feat, seqLengths, rnnModule, rnnHiddenSize)
    local fwdLstm = nn.MaskRNN(rnnModule:clone())({ feat, seqLengths })
    local bwdLstm = nn.ReverseMaskRNN(rnnModule:clone())({ feat, seqLengths })
    local rnn = nn.Sequential():add(nn.CAddTable())
    rnn:add(nn.BatchNormalization(rnnHiddenSize, 1e-3))
    return rnn({ fwdLstm, bwdLstm })
end


-- Based on convolution kernel and strides.
local function calculateInputSizes(sizes)
    sizes = torch.floor((sizes - 8) / 2 + 1) -- conv1
    sizes = torch.floor((sizes - 8) / 2 + 1) -- conv2
--    sizes = torch.floor((sizes - 2) / 2 + 1) -- pool1
    return sizes
end


local function get_min_width()
    local width = 1
--    width = (width+1) * 2 + 2
    width = (width+1) * 2 + 8
    width = (width+1) * 2 + 8
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
    local seqLengths = nn.Identity()()
    local input = nn.Identity()()
    local feature = nn.Sequential()

    -- (nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]) conv layers.
    feature:add(nn.SpatialConvolution(1, 32, 8, 12, 2, 2))
    feature:add(nn.SpatialBatchNormalization(32, 1e-3))
    feature:add(nn.ReLU(true))
    feature:add(nn.SpatialConvolution(32, 32, 8, 2, 2, 1))
    feature:add(nn.SpatialBatchNormalization(32, 1e-3))
    feature:add(nn.ReLU(true))
    -- TODO the DS2 architecture does not include this layer, but mem overhead increases.
    --feature:add(nn.SpatialMaxPooling(2, 2, 2, 2)) 

    local tmp = torch.rand(1, 1, height, get_min_width()*2) --init something
    local rnnInputsize = 32 * feature:forward(tmp):size(3) -- outputPlanes X outputHeight
    local rnnHiddenSize = 400 -- size of rnn hidden layers
    local nbOfHiddenLayers = 4

    feature:add(nn.View(rnnInputsize, -1):setNumInputDims(3)) -- batch x features x seqLength
    feature:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- seqLength x batch x features
    feature:add(nn.View(-1, rnnInputsize)) -- (seqLength x batch) x features

    local rnn = nn.Identity()({ feature(input) })
    local rnn_module = getRNNModule(rnnInputsize, rnnHiddenSize,
                                    GRU, isCUDNN)
    rnn = BRNN(rnn, seqLengths, rnn_module, rnnHiddenSize)
    rnn_module = getRNNModule(rnnHiddenSize, rnnHiddenSize,
                              GRU, isCUDNN)

    for i = 1, nbOfHiddenLayers do
        rnn = BRNN(rnn, seqLengths, rnn_module, rnnHiddenSize)
    end

    local post_sequential = nn.Sequential()
    post_sequential:add(nn.Linear(rnnHiddenSize, dict_size))
    local model = nn.gModule({ input, seqLengths }, { post_sequential(rnn) })
    model = makeDataParallel(model, nGPU, isCUDNN)
    return model
end


return { deepSpeech, calculateInputSizes, get_min_width }
