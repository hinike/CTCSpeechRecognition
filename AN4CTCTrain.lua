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
    nGPU = 1, -- Number of GPUs, set -1 to use CPU
    
    -- these 5 usually need to change together
    modelName = 'DeepSpeechModelSpect',
    trainingSetLMDBPath = './prepare_an4_spect/train/',-- online loading path data.
    validationSetLMDBPath = './prepare_an4_spect/test/',
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
    dictSize = 28,

    trainIteration= 47*70,
    batchSize = 20,
    validationBatchSize = 20,
    validationIterations = 7,
    testGap = 47,
    saveModelIterations = 47*20, -- iterations! Intead of Epoch
    
}
--Parameters for the stochastic gradient descent (using the optim library).
local sgdParams = {
    learningRate = 1e-3,
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
