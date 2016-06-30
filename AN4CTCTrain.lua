--[[Trains the CTC model using the AN4 audio database.]]

local Network = require 'Network'

--Training parameters
torch.setdefaulttensortype('torch.FloatTensor')
seed = 10
torch.manualSeed(seed)
cutorch.manualSeedAll(seed)

local networkParams = {
    loadModel = false,
    saveModel = true,
    backend = 'cudnn',
    nGPU = 4, -- Number of GPUs, set -1 to use CPU

    -- these 5 usually need to change together
    modelName = 'DeepSpeechModelSpect',
    trainingSetLMDBPath = '/data1/zhirongw/LibriSpeech/train',-- online loading path data.
    validationSetLMDBPath = '/data1/zhirongw/LibriSpeech/test/',
    feature = 'spect', -- can be spect or logfbank
    dataHeight = 129, -- if using logfbank, this means nfilts

--    modelName = 'DeepSpeechModelLogFBank',
--    trainingSetLMDBPath = './prepare_an4/train/',-- online loading path data.
--    validationSetLMDBPath = './prepare_an4/test/',
--    feature = 'logfbank', -- can be spect or logfbank
--    dataHeight = 26, -- if using logfbank, this means nfilts

    logsTrainPath = './logs/TrainingLoss/',
    logsValidationPath = './logs/ValidationScores/',
    modelTrainingPath = './models/',
    fileName = arg[1] or 'CTCNetwork.t7',

    dictionaryPath = './dictionary',
    dictSize = 29,

    epochs = 70,
    batchSize = 200,
    validationBatchSize = 24,
    validationIterations = 100,
    saveModelIterations = 10, -- Epochs

}
--Parameters for the stochastic gradient descent (using the optim library).
local sgdParams = {
    learningRate = 1e-1,
    learningRateDecay = 1e-9,
    weightDecay = 0,
    momentum = 0.9,
    dampening = 0,
    nesterov = true
}

--Create and train the network based on the parameters and training data.
Network:init(networkParams)

Network:trainNetwork(sgdParams)

--Creates the loss plot.
Network:createLossGraph()

print("finished")
