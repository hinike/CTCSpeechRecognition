require 'optim'
require 'nnx'
require 'gnuplot'
require 'lfs'
require 'xlua'
require 'UtilsMultiGPU'
require 'Loader'
require 'nngraph'
require 'Mapper'
require 'WEREvaluator'

local suffix = '_' .. os.date('%Y%m%d_%H%M%S')
local threads = require 'threads'
local Network = {}

function Network:init(networkParams)

    self.fileName = networkParams.fileName -- The file name to save/load the network from.
    self.nGPU = networkParams.nGPU
    self.isCUDNN = networkParams.backend == 'cudnn'
    if self.nGPU <= 0 then
        assert(not self.isCUDNN)
    end
    assert(networkParams.batchSize % networkParams.nGPU == 0, 'batch size must be the multiple of nGPU')
    assert(networkParams.validationBatchSize % networkParams.nGPU == 0, 'batch size must be the multiple of nGPU')
    self.trainingSetLMDBPath = networkParams.trainingSetLMDBPath
    self.validationSetLMDBPath = networkParams.validationSetLMDBPath
    self.logsTrainPath = networkParams.logsTrainPath or nil
    self.logsValidationPath = networkParams.logsValidationPath or nil
    self.modelTrainingPath = networkParams.modelTrainingPath or nil
    self.trainEpochs = networkParams.epochs

    self.dataHeight = networkParams.dataHeight
    self.feature = networkParams.dataHeight

    self:makeDirectories({ self.logsTrainPath, self.logsValidationPath, self.modelTrainingPath })

    self.mapper = Mapper(networkParams.dictionaryPath)
    self.saveModel = networkParams.saveModel
    self.loadModel = networkParams.loadModel
    self.saveModelIterations = networkParams.saveModelIterations or 1000 -- Saves model every number of iterations.

    -- TODO may not work for current version
    -- setting model saving/loading
    if (self.loadModel) then
        assert(networkParams.fileName, "Filename hasn't been given to load model.")
        self:loadNetwork(networkParams.fileName,
            networkParams.modelName,
            self.isCUDNN)
    else
        assert(networkParams.modelName, "Must have given a model to train.")
        self:prepSpeechModel(networkParams.modelName, networkParams.dataHeight,
            networkParams.dictSize)
    end
    --assert((networkParams.saveModel or networkParams.loadModel) and
    --    networkParams.fileName, "To save/load you must specify the fileName you want to save to")

    self.werTester = WEREvaluator(self.validationSetLMDBPath, self.mapper,
        networkParams.validationBatchSize, networkParams.validationIterations,
        self.logsValidationPath, networkParams.feature, networkParams.dataHeight,
        networkParams.modelName)

    self.logger = optim.Logger(self.logsTrainPath .. 'train' .. suffix .. '.log')
    self.logger:setNames { 'loss', 'WER' }
    self.logger:style { '-', '-' }

    self.trainLoader = Loader(networkParams.trainingSetLMDBPath,
         networkParams.batchSize, networkParams.feature,
         networkParams.dataHeight, networkParams.modelName)
    self.trainLoader.lmdb_size = 132400
    --self.trainLoader:prep_sorted_inds()
end


function Network:prepSpeechModel(modelName, dataHeight, dict_size)
    local model = require(modelName)
    self.model = model[1](self.nGPU, self.isCUDNN, dataHeight, dict_size)
    self.calSizeOfSequences = model[2]
end

local function replace(self, callback)
  local out = callback(self)
  if self.modules then
    for i, module in ipairs(self.modules) do
      self.modules[i] = replace(module, callback)
    end
  end
  return out
end

local function convertBN(net, dst)
    return replace(net, function(x)
        local y = 0
        local src = dst == nn and cudnn or nn
        local src_prefix = src == nn and 'nn.' or 'cudnn.'
        local dst_prefix = dst == nn and 'nn.' or 'cudnn.'
        -- print (torch.typename(x), src_prefix..'BatchNormalization')

        local function convert(v)
            local y = {}
            torch.setmetatable(y, dst_prefix..v)
            for k,u in pairs(x) do y[k] = u end
            if src == cudnn and x.clearDesc then x.clearDesc(y) end
            -- print (v,' ',y)
            return y
        end
        if torch.typename(x) == src_prefix..'BatchNormalization' then
            -- print (x)
            y = convert('BatchNormalization')
        end
        return y == 0 and x or y
    end)
end

function Network:testNetwork()
    self.model:evaluate()
    require 'BNDecorator'
    if self.isCUDNN then
        if self.nGPU > 1 then
            self.model.impl:exec(function(m, i)
                convertBN(m, nn)
            end)
        else
            self.model = convertBN(self.model, nn)
        end
    end
    local results = self.werTester:getWER(self.nGPU > 0, self.model, self.calSizeOfSequences, false) -- details in log
    self.model:zeroGradParameters()
    if self.isCUDNN then
        if self.nGPU > 1 then
            self.model.impl:exec(function(m, i)
                convertBN(m, cudnn)
            end)
        else
            self.model = convertBN(self.model, cudnn)
        end
    end
    self.model:training()
    return results
end


function Network:trainNetwork(sgd_params)
    --[[
        train network with self-defined feval (sgd inside); use ctc for evaluation
    --]]
    self.model:training()

    local lossHistory = {}
    local validationHistory = {}

    local x, gradParameters = self.model:getParameters()

    local criterion
    if self.nGPU <= 1 then
        criterion = nn.CTCCriterion()
    end
    if self.nGPU > 0 then
        if self.nGPU == 1 then
            criterion = criterion:cuda()
        end
    end

    -- define the feval
    local loss
    local function feval(x_new)
        return loss, gradParameters
    end

    local dataTimer = torch.Timer()
    local timer = torch.Timer()
    -- training
    local startTime = os.time()
    local averageLoss = 0

    for i = 1, self.trainEpochs do
        for n, sample in self.trainLoader:nxt_batch(self.trainLoader.DEFAULT) do
            --------------------- data load ------------------------
            local datatime = dataTimer:time().real
            local inputs, sizes, targets, labelcnt

            sizes = self.calSizeOfSequences(sample.sizes)
            targets = sample.label
            labelcnt = sample.labelcnt
            if self.nGPU > 0 then
                inputs = sample.inputs:cuda()
                sizes = sizes:cuda()
            end
            --------------------- fwd and bwd ---------------------
            if criterion then
                local output = self.model:forward(inputs)
                criterion:forward(output, targets, sizes)
                loss = criterion.output
                local gradOutput = criterion:backward(output, targets)
                self.model:zeroGradParameters()
                self.model:backward(inputs, gradOutput)
            else
                self.model:forward(inputs)
                self.model:zeroGradParameters()
                loss = self.model:backward(inputs, targets, sizes)
            end
            --gradParameters:div(inputs:size(1))
            gradParameters:div(labelcnt)
            loss = loss / labelcnt
            gradParameters:clamp(-1,1)
            
            local _, fs = optim.sgd(feval, x, sgd_params)
            averageLoss = 0.9 * averageLoss + 0.1 * fs[1]

            local itertime = timer:time().real
            print(('Iter: [%d][%d]. Time %.3f data %.3f Ratio %.3f. Error: %1.3f. Learning rate: %.3f')
                :format(i, n, itertime, datatime, datatime/itertime, fs[1], sgd_params.learningRate))

            timer:reset()
            dataTimer:reset()
        end
        -- Testing
        local results = self:testNetwork()
        print(('TESTING EPOCH: [%d]. Loss: %1.3f WER: %2.2f%%.'):format(i, results.loss, results.wer * 100))

        table.insert(lossHistory, averageLoss) -- Add the average loss value to the logger.
        table.insert(validationHistory, 100 * wer)
        self.logger:add { averageLoss, 100 * wer }

        -- snapshot the model
        if self.saveModel and i % self.saveModelIterations == 0 then
            print("Saving model..")
            self:saveNetwork(self.modelTrainingPath .. '_epoch_' .. j ..
                suffix .. '_' .. self.fileName)
        end
    end

    local endTime = os.time()
    local secondsTaken = endTime - startTime
    local minutesTaken = secondsTaken / 60
    print("Minutes taken to train: ", minutesTaken)

    if self.saveModel then
        print("Saving model..")
        self:saveNetwork(self.modelTrainingPath .. 'final_model' .. suffix .. '.t7')
    end

    return lossHistory, validationHistory, minutesTaken
end

function Network:createLossGraph()
    self.logger:plot()
end

function Network:saveNetwork(saveName)
    saveDataParallel(saveName, self.model)
end

--Loads the model into Network.
function Network:loadNetwork(saveName, modelName)
    print ('loading model ' .. saveName)
    local model
    self.prepSpeechModel(model, modelName)
    local weights, gradParameters = self.model:getParameters()
    model = loadDataParallel(saveName, self.nGPU, self.isCUDNN)
    local weights_to_copy, _ = model:getParameters()
    weights:copy(weights_to_copy)
end

function Network:makeDirectories(folderPaths)
    for index, folderPath in ipairs(folderPaths) do
        if (folderPath ~= nil) then os.execute("mkdir -p " .. folderPath) end
    end
end

return Network
