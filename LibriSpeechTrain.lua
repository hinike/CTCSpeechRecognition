--[[Trains the CTC model using the AN4 audio database.]]

local Network = require 'Network'

--Training parameters
torch.setdefaulttensortype('torch.FloatTensor')
seed = 10
torch.manualSeed(seed)
cutorch.manualSeedAll(seed)

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Deep Speech Model')
cmd:text()
cmd:text('Options')

-- generals
cmd:option('-nGPU',4,'number of GPU to use.')
cmd:option('backend', 'cudnn', 'use nn or cudnn')
cmd:option('saveModel', true, 'save model?')
cmd:option('loadModel', false, 'load model?')
-- paths: data and logs
cmd:option('-trainingSetLMDBPath', '/data1/zhirongw/LibriSpeech/train/', 'training lmdb')
cmd:option('-validationSetLMDBPath', '/data1/zhirongw/LibriSpeech/test/', 'validation lmdb')
cmd:option('-logsTrainPath', './logs/TrainingLoss/', 'loss log directory')
cmd:option('-logsValidationPath', './logs/ValidationScores/', 'scores log directory')
cmd:option('-modelTrainingPath', './models/', 'model snapshot directory')
cmd:option('-fileName', 'CTCNetwork.t7', 'model snapshot filename')
cmd:option('-dictionaryPath', './dictionary', 'dictionary path')
-- Model
cmd:option('-feature', 'spect', 'input feature of the sound wave')
cmd:option('-dataHeight', 129, 'feature dimension')
cmd:option('-dictSize', 29, 'language dictionary size')
cmd:option('-modelName', 'DeepSpeechModelSpect', 'model architecture')
cmd:option('-hiddenSize', 400, 'LSTM hidden memory size')
cmd:option('-num_layers', 5, 'number of LSTM layers')
-- configs
cmd:option('-batchSize', 200, 'training batch size')
cmd:option('-epochs', 70, 'training epochs')
cmd:option('-validationBatchSize', 24, 'testing batch size')
cmd:option('-saveModelIterations', 10, 'every several epochs for snapshot')
-- optim
cmd:option('-optim','sgd','optimization algorithm')
cmd:option('-learning_rate',1e-1,'learning rate')
cmd:option('-learning_rate_decay',1e-9,'learning rate decay')
cmd:option('-learning_rate_decay_after',20,'in number of epochs, when to start decaying the learning rate')
cmd:option('-learning_rate_decay_every',20,'decrease learning rate every')
cmd:option('-beta1',0.8,'beta1 for adam')
cmd:option('-beta2',0.95,'beta2 for adam')
cmd:option('-alpha',0.8,'alpha for rmsprop')
cmd:option('-weight_decay',0,'weight decay')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-grad_clip',1,'clip gradients at this value') -- not used
cmd:text()

local opts = cmd:parse(arg)

--Parameters for the stochastic gradient descent (using the optim library).
local sgdParams = {
    learningRate = opts.learning_rate,
    learningRateDecay = opts.learning_rate_decay,
    weightDecay = 0,
    momentum = 0.9,
    dampening = 0,
    nesterov = true
}

--Create and train the network based on the parameters and training data.
Network:init(opts)

Network:trainNetwork()

--Creates the loss plot.
Network:createLossGraph()

print("finished")
