require 'Loader'
require 'Util'
require 'Mapper'
require 'torch'
require 'xlua'
require 'cutorch'
local threads = require 'threads'
local Evaluator = require 'Evaluator'

local WEREvaluator = torch.class('WEREvaluator')

function WEREvaluator:__init(_path, mapper, testBatchSize, 
    nbOfTestIterations, logsPath, feature, dataHeight, modelname)

    self.testBatchSize = testBatchSize
    self.nbOfTestIterations = nbOfTestIterations
    self.feature = feature

    self.mapper = mapper
    self.logsPath = logsPath
    self.suffix = '_' .. os.date('%Y%m%d_%H%M%S')

    self.testLoader = Loader(_path, testBatchSize, feature, dataHeight, modelname)
end

function WEREvaluator:predicTrans(src, nGPU)
    local gpu_number = nGPU or 1
    return src:view(-1, self.testBatchSize / gpu_number, src:size(2)):transpose(1,2)
end

function WEREvaluator:getWER(gpu, model, calSizeOfSequences, verbose, currentIteration)
    --[[
        load test_iter*batch_size data point from test set; compute average WER

        input:
            verbose:if true then print WER and predicted strings for each data to log
    --]]

    local cumWER = 0
    local inputs = torch.Tensor()
    if (gpu) then
        inputs = inputs:cuda()
    end
    local specBuf, labelBuf, sizesBuf

    if verbose then
        local f = assert(io.open(self.logsPath .. 'WER_Test' .. self.suffix .. '.log', 'a'),
               "Could not create validation test logs, does the folder "
                .. self.logsPath .. " exist?")
        f:write('======================== BEGIN WER TEST currentIteration: '
                .. currentIteration .. ' =========================\n')
        f:close()
    end

    local werPredictions = {} -- stores the predictions to order for log.
    local N = 0
    -- ======================= for every test iteration ==========================
    for n, sample in self.testLoader:nxt_batch() do
        -- get buf and fetch next one
        local inputs, sizes, targets, labelcnt

        sizes = calSizeOfSequences(sample.sizes)
        targets = sample.label
        labelcnt = sample.labelcnt
        if gpu then
            inputs = sample.inputs:cuda()
            sizes = sizes:cuda()
        end
        local predictions = model:forward(inputs)
        if type(predictions) == 'table' then
            local temp = self:predicTrans(predictions[1], #predictions)
            for k = 2, #predictions do
                temp = torch.cat(temp, self:predicTrans(predictions[k], #predictions), 1)
            end
            predictions = temp
        else
            predictions = self:predicTrans(predictions)
        end

        -- =============== for every data point in this batch ==================
        local batchWER = 0
        for j = 1, self.testBatchSize do
            local prediction_single = predictions[j]
            local predict_tokens = Evaluator.predict2tokens(prediction_single, self.mapper)
            local WER = Evaluator.sequenceErrorRate(targets[j], predict_tokens)
            cumWER = cumWER + WER
            batchWER = batchWER + WER
            table.insert(werPredictions, { wer = WER * 100, target = self:tokens2text(targets[j]), prediction = self:tokens2text(predict_tokens) })
        end
        print(('Testing | Iter: %d, WER: %2.2f%%'):format(n, batchWER/self.testBatchSize*100))
        N = N + 1
    end

    local function comp(a, b) return a.wer < b.wer end

    table.sort(werPredictions, comp)

    if verbose then
        for index, werPrediction in ipairs(werPredictions) do
            local f = assert(io.open(self.logsPath .. 'WER_Test' .. self.suffix .. '.log', 'a'))
            f:write(string.format("WER = %.2f%% | Text = \"%s\" | Predict = \"%s\"\n",
                werPrediction.wer, werPrediction.target, werPrediction.prediction))
            f:close()
        end
    end
    local averageWER = cumWER / (N * self.testBatchSize)
    local f = assert(io.open(self.logsPath .. 'WER_Test' .. self.suffix .. '.log', 'a'))
    f:write(string.format("Average WER = %.2f%%", averageWER * 100))
    f:close()

    return averageWER
end

function WEREvaluator:tokens2text(tokens)
    local text = ""
    for i, t in ipairs(tokens) do
        text = text .. self.mapper.token2alphabet[tokens[i]]
    end
    return text
end
