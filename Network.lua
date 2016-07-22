require 'optim'
require 'nnx'
require 'gnuplot'
require 'lfs'
require 'xlua'
require 'UtilsMultiGPU'
require 'Loader'
require 'Mapper'
require 'WEREvaluator'

local suffix = '_' .. os.date('%Y%m%d_%H%M%S')
local threads = require 'threads'
local Network = {}

function Network:init(networkParams)

    self.opts = networkParams

    self.opts.isCUDNN = self.opts.backend == 'cudnn'
    if self.opts.nGPU <= 0 then
        assert(not self.opts.isCUDNN)
    end
    assert(self.opts.batchSize % self.opts.nGPU == 0, 'batch size must be the multiple of nGPU')
    assert(self.opts.validationBatchSize % self.opts.nGPU == 0, 'batch size must be the multiple of nGPU')

    self:makeDirectories({ self.opts.logsTrainPath, self.opts.logsValidationPath, self.opts.modelTrainingPath })

    -- TODO may not work for current version
    -- setting model saving/loading
    if (self.opts.loadModel) then
        assert(self.opts.fileName, "Filename hasn't been given to load model.")
        self:loadNetwork(self.opts.fileName,
                         self.opts.modelName,
                         self.opts.isCUDNN)
    else
        assert(self.opts.modelName, "Must have given a model to train.")
        self:prepSpeechModel()
    end

    -- setting online loading

    self.werTester = WEREvaluator(self.opts.validationSetLMDBPath,
                                  Mapper(self.opts.dictionaryPath),
                                  self.opts.validationBatchSize,
                                  self.opts.validationIterations,
                                  self.opts.logsValidationPath,
                                  self.opts.feature, self.opts.dataHeight,
                                  self.opts.modelName)

    self.logger = optim.Logger(self.opts.logsTrainPath .. 'train' .. suffix .. '.log')
    self.logger:setNames { '    loss    ', '    WER' }
    self.logger:style { '-', '-' }

    self.trainLoader = Loader(self.opts.trainingSetLMDBPath,
                              self.opts.batchSize, 
                              self.opts.feature,
                              self.opts.dataHeight,
                              self.opts.modelName)
    self.trainLoader.lmdb_size = 132400
    --self.trainLoader:prep_sorted_inds()
end

function Network:prepSpeechModel()
    local model = require(self.opts.modelName)
    self.model = model[1](self.opts.rnn_type,
                          self.opts.hidden_size,
                          self.opts.num_layers,
                          self.opts.dictSize,
                          self.opts.nGPU,
                          self.opts.isCUDNN)
    self.calSizeOfSequences = model[2]
end

local function convertBN(net, dst)
    local toCUDNN = not (dst == nn)
    local function rpl(x)
        local y = 0
        local src_prefix = toCUDNN and 'nn.' or 'cudnn.'
        local dst_prefix = toCUDNN and 'cudnn.' or 'nn.'
    --    print (torch.typename(x), src_prefix..'BatchNormalization')

        local function convert(v)
            local y = {}
            torch.setmetatable(y, dst_prefix..v)
            for k,u in pairs(x) do y[k] = u end
            if (not toCUDNN) and x.clearDesc then x.clearDesc(y) end
            return y
        end
        if torch.typename(x) == src_prefix..'BatchNormalization' then
            y = convert('BatchNormalization')
        end
        return y == 0 and x or y
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

    local function impl_rpl(m)
        replace(m, rpl)
    end

    if net.opts.isCUDNN then
        if net.opts.nGPU > 1 then
            net.model.impl:exec(impl_rpl)
        else
            net.model = replace(net.model, rpl)
        end
    end
end

function Network:testNetwork(currentIteration)
    self.model:evaluate()
    convertBN(self, nn)
    local results = self.werTester:getWER(self.opts.nGPU > 0, self.model, self.calSizeOfSequences, true, currentIteration) -- details in log
    self.model:zeroGradParameters()
    convertBN(self, cudnn)
    self.model:training()
    return results
end


function Network:trainNetwork()
    --[[
        train network with self-defined feval (sgd inside); use ctc for evaluation
    --]]
    self.model:training()

    local lossHistory = {}
    local validationHistory = {}

    local x, gradParameters = self.model:getParameters()
    print('Number of network parameters: ' .. x:nElement())

    local criterion
    if self.opts.nGPU <= 1 then
        criterion = nn.CTCCriterion()
    end
    if self.opts.nGPU == 1 then
        criterion = criterion:cuda()
    end

    local optim_params = {
        beta1 = self.opts.beta1, -- adam
        beta2 = self.opts.beta2, -- adam
        alpha = self.opts.alpha, -- rmsprop
        weightDecay = 0,
        momentum = 0.9,
        dampening = 0,
        nesterov = true, -- for sgd
    }
    -- define the feval
    local loss
    local function feval(x_new)
        return loss, gradParameters
    end

    -- training
    local dataTimer = torch.Timer()
    local timer = torch.Timer()
    local averageLoss = 0

    for i = 1, self.opts.epochs do
--	  local batch_type = i < 40 and self.trainLoader.DEFAULT or self.trainLoader.RANDOM
        local batch_type = self.trainLoader.RANDOM
        for n, sample in self.trainLoader:nxt_batch(batch_type) do
            --------------------- data load ------------------------
            local datatime = dataTimer:time().real
            local inputs, sizes, targets, labelcnt

            sizes = self.calSizeOfSequences(sample.sizes)
            targets = sample.label
            labelcnt = sample.labelcnt
            if self.opts.nGPU > 0 then
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
            gradParameters:div(inputs:size(1))
            --gradParameters:div(labelcnt)
            loss = loss / labelcnt
            gradParameters:clamp(-0.1,0.1)

            optim_params.learningRate = self:LearningRate(i)
            local fs
            if self.opts.optim == 'sgd' then
                _, fs = optim.sgd(feval, x, optim_params)
            elseif self.opts.optim == 'rmsprop' then
                _, fs = optim.rmsprop(feval, x, optim_params)
            elseif self.opts.optim == 'adam' then
                _, fs = optim.adam(feval, x, optim_params)
            end
            averageLoss = 0.9 * averageLoss + 0.1 * fs[1]

            local itertime = timer:time().real
            print(('Iter: [%d][%d]. Time %.3f data %.3f Ratio %.3f. Error: %1.3f. Learning rate: %f')
                :format(i, n, itertime, datatime, datatime/itertime, fs[1], optim_params.learningRate))

            timer:reset()
            dataTimer:reset()
        end
        -- Testing
--	if i % 50 == 0 then
        local results = self:testNetwork(i)
        print(('TESTING EPOCH: [%d]. Loss: %1.3f WER: %2.2f%%.'):format(i, results.loss, results.WER * 100))

        table.insert(lossHistory, averageLoss) -- Add the average loss value to the logger.
        table.insert(validationHistory, 100 * results.WER)
        self.logger:add { averageLoss, 100 * results.WER }
	end
        -- snapshot the model
        if self.opts.saveModel and i % self.opts.saveModelIterations == 0 then
            print("Saving model..")
            self:saveNetwork(self.opts.modelTrainingPath .. '_epoch_' .. i ..
                suffix .. '_' .. self.opts.fileName)
        end
  --  end

    if self.saveModel then
        print("Saving model..")
        self:saveNetwork(self.opts.modelTrainingPath .. 'final_model' .. suffix .. '.t7')
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
    model = loadDataParallel(saveName, self.opts.nGPU, self.opts.isCUDNN)
    local weights_to_copy, _ = model:getParameters()
    weights:copy(weights_to_copy)
end

function Network:makeDirectories(folderPaths)
    for index, folderPath in ipairs(folderPaths) do
        if (folderPath ~= nil) then os.execute("mkdir -p " .. folderPath) end
    end
end

function Network:LearningRate(epoch)
   -- Training schedule
   local decay = math.floor((epoch - self.opts.learning_rate_decay_after - 1) / self.opts.learning_rate_decay_every) + 1
   return self.opts.learning_rate * math.pow(0.1, decay)
end

return Network
