require 'nn'
require 'torch'
require 'lmdb'
require 'xlua'
require 'paths'
tds = require 'tds'
threads = require 'threads'

-- get model specified methods into this module
local model_t
local cal_size
local get_min_width

local util = require 'Util'

--[[

    this file defines Loader and loader:
        - Loader returns different inds of nxt btach
        - loader loads data from lmdb given the inds

    NOTE:
        - make sure calculateInputSizes() in DeepSpeechModel.lua is set correctly

--]]

torch.setdefaulttensortype('torch.FloatTensor')

local Loader = torch.class('Loader')

function Loader:__init(_dir, batch_size, feature, dataHeight, modelname)
    --[[
        input:
            feature: is it spect or logfbank we are using
            dataHeight: typically 129 for spect; 26 for logfbank
    --]]

    -- constants to indicate the loading style
    self.DEFAULT = 1
    self.SAMELEN = 2
    self.SORTED = 3

    self.modelname = modelname
    self.batch_size = batch_size
    self.dataHeight = dataHeight
    self.is_spect = feature == 'spect'
    self.cnt = 1

    self.sorted_inds = {}
    self.len_num = 0 -- number of unique seqLengths
    --self.min_width = get_min_width() --from DeepSpeech

    local function preprocess()
        -- assume the super folder is the lmdb root folder
        local lmdb_path = _dir..'/../'
        local stats = {} -- mean/std
        --print('preparing mean and std of the dataset..')
        if paths.filep(lmdb_path..'mean_std') then
            --print('found previously saved stats..')
            stats = torch.load(lmdb_path..'mean_std')
        else
            print('did not find previously saved stats, generating..')
            if feature == 'spect' then
                util.get_mean_std(lmdb_path)
            else
                util.get_mean_std(lmdb_path, dataHeight)
            end
            stats = torch.load(lmdb_path..'mean_std')
        end
        return stats
    end

    local function init() require('Mapper') require('lmdb') tds = require 'tds' end

    local function main(idx)
        torch.manualSeed(idx)
        torch.setnumthreads(1)
        -- get model specified methods
        model_t = require(modelname)
        _G.cal_size = model_t[2]
        _G.get_min_width = model_t[3]

        _G.db_spect = lmdb.env { Path = _dir .. '/spect', Name = 'spect' }
        _G.db_label = lmdb.env { Path = _dir .. '/label', Name = 'label' }
        _G.db_trans = lmdb.env { Path = _dir .. '/trans', Name = 'trans' }

        -- get the size of lmdb
        _G.db_spect:open()
        _G.db_label:open()
        _G.db_trans:open()
        local l1 = _G.db_spect:stat()['entries']
        local l2 = _G.db_label:stat()['entries']
        local l3 = _G.db_trans:stat()['entries']

        assert(l1 == l2 and l2 == l3, 'data sizes in each lmdb must agree')

        _G.db_spect:close()
        _G.db_label:close()
        _G.db_trans:close()

        _G.stats = preprocess()
        return l1
    end
    
    local pool, lmdb_size = threads.Threads(1, init, main)

    self.pool = pool
    self.lmdb_size = lmdb_size[1][1]
end

function Loader:prep_sorted_inds()
    --[[
        prep a table for sorted inds, can detect previously saved table in lmdb folder
    --]]


    print('preparing sorted indices..')
    local indicesFilePath = self._dir .. '/' .. 'sorted_inds_' .. self.min_width

    -- check if there is previously saved inds
    if paths.filep(indicesFilePath) then
        print('found previously saved inds..')
        self.sorted_inds = torch.load(indicesFilePath)
        print('original size: '..self.lmdb_size..' valid data: '..#self.sorted_inds)
        self.lmdb_size = #self.sorted_inds
        return
    end

    -- if not make a new one
    print('did not find previously saved indices, generating.')

    model_t = require(modelname)
    cal_size = model_t[2]
    get_min_width = model_t[3]

    self.db_spect = lmdb.env { Path = _dir .. '/spect', Name = 'spect' }
    self.db_label = lmdb.env { Path = _dir .. '/label', Name = 'label' }
    self.db_trans = lmdb.env { Path = _dir .. '/trans', Name = 'trans' }

    self.db_spect:open(); local txn = self.db_spect:txn(true)
    self.db_label:open(); local txn_label = self.db_label:txn(true)
    self.lmdb_size = self.db_spect:stat()['entries']

    local lengths = {}
    -- those shorter than min_width are ignored
    local true_size = 0
    for i = 1, self.lmdb_size do

        local lengthOfAudio
        if self.is_spect then
            lengthOfAudio = txn:get(i):size(2)
        else
            lengthOfAudio = txn:get(i, true):size(1) / (4*self.dataHeight)
        end

        local lengthOfLabel = #(torch.deserialize(txn_label:get(i)))

        if lengthOfAudio >= self.min_width and cal_size(lengthOfAudio) >= lengthOfLabel then
            true_size = true_size + 1
            table.insert(self.sorted_inds, { i, lengthOfAudio })
            if lengths[lengthOfAudio] == nil then lengths[lengthOfAudio] = true end
            if i % 100 == 0 then xlua.progress(i, self.lmdb_size) end
        end
    end

    print('original size: '..self.lmdb_size..' valid data: '..true_size)
    self.lmdb_size = true_size -- set size to true size
    txn:abort(); self.db_spect:close()
    txn_label:abort(); self.db_label:close()

    local function comp(a, b) return a[2] < b[2] end
    table.sort(self.sorted_inds, comp)

    for _ in pairs(lengths) do self.len_num = self.len_num + 1 end -- number of different seqLengths
    torch.save(indicesFilePath, self.sorted_inds)
end


function Loader:nxt_sorted_inds()
    local meta_inds = self:nxt_inds()
    local inds = meta_inds:clone()
    for i = 1, size(inds,1) do
        inds[i]  = self.sorted_inds[meta_inds[i]][1]
    end
    return inds
end


function Loader:nxt_same_len_inds()
    --[[
        return inds with same seqLength, a solution before zero-masking can work
    --]]

    local _len = self.sorted_inds[self.cnt][2]
    while (self.cnt <= self.lmdb_size and self.sorted_inds[self.cnt][2] == _len) do
        -- NOTE: true index store in table, instead of cnt
        table.insert(inds, self.sorted_inds[self.cnt][1])
        self.cnt = self.cnt + 1
    end

    if self.cnt > self.lmdb_size then self.cnt = 1 end

    return inds
end

function Loader:nxt_inds()
    --[[
        return indices of the next batch
    --]]

    local inds = torch.linspace(self.cnt, self.cnt+self.batch_size-1, self.batch_size)
    self.cnt = self.cnt + self.batch_size
    local overflow = inds[-1] - self.lmdb_size
    if overflow > 0 then
        inds:narrow(1, self.batch_size-overflow+1, overflow):copy(torch.linspace(1, overflow, overflow))
        self.cnt = overflow + 1
    end
    return inds
end

function Loader:convert_tensor(btensor)
    --[[
        convert a 1d byte tensor to 2d float tensor.
    --]]

    local num = btensor:size(1) / 4 -- assume real data is float
    local s = torch.FloatStorage(num, tonumber(torch.data(btensor, true)))

    assert(num % self.dataHeight == 0, 'something wrong with the tensor dims')

    return torch.FloatTensor(s, 1, torch.LongStorage{self.dataHeight, num / self.dataHeight})

end

function Loader:nxt_batch(mode)
    --[[
        return a batch by loading from lmdb just-in-time

        input:
            mode: should be Loader.DEFAULT/SAMELEN/SORTED; USE ONLY ONE MODE FOR ONE TRAINING
            flag: indicates whether to load trans

        TODO we allocate 2 * batch_size space
    --]]

    local pool = self.pool

    local idx, sample = 1, nil
    local function enqueue()
        while idx <= self.lmdb_size and pool:acceptsjob() do
            -- gen index for this iter batch
            local indices
            if mode == self.SAMELEN then
                assert(#self.sorted_inds > 0, 'call prep_sorted_inds before nxt_batch')
                indices = self:nxt_same_len_inds()
            elseif mode == self.SORTED then
                assert(#self.sorted_inds > 0, 'call prep_sorted_inds before nxt_batch')
                indices = self:nxt_sorted_inds()
            else -- default
                indices = self:nxt_inds()
            end
            pool:addjob(
                function(indices)
                    local tensor_list = tds.Vec()
                    local label_list = {}
                    local sizes_array = torch.Tensor(#indices)
                    local labelcnt = 0

                    local max_w = 0
                    local h = 0
                  
                    _G.db_spect:open(); local txn_spect = _G.db_spect:txn(true) -- readonly
                    _G.db_label:open(); local txn_label = _G.db_label:txn(true)

                    is_spect = true
                    for i, idx in ipairs(indices:totable()) do
                       local tensor
                       if is_spect then
                           tensor = txn_spect:get(idx)
                       else
                    --       tensor = self:convert_tensor(txn_spect:get(idx, true))
                       end

                       local label = torch.deserialize(txn_label:get(idx))

                       h = tensor:size(1)
                       sizes_array[i] = tensor:size(2) 
                       if max_w < tensor:size(2) then max_w = tensor:size(2) end -- find the max len in this batch

                       tensor_list:insert(tensor)
                       table.insert(label_list, label)
                       labelcnt = labelcnt + #label
                   end
                   -- store tensors into a fixed len tensor_array TODO should find a better way to do this
                   local tensor_array = torch.Tensor(indices:size(1), 1, h, max_w):zero()
                   for i, tensor in ipairs(tensor_list) do
                       tensor_array[i][1]:narrow(2, 1, tensor:size(2)):copy(tensor)
                   end
                   tensor_array:csub(_G.stats[1])
                   tensor_array:div(_G.stats[2])

                   txn_spect:abort(); _G.db_spect:close()
                   txn_label:abort(); _G.db_label:close()
                   return {
                       inputs = tensor_array,
                       label = label_list,
                       sizes = sizes_array,
                       labelcnt = labelcnt,
                   }
                end,
                function(_sample_)
                    sample = _sample_
                end,
                indices)
            idx = idx + indices:size(1)
        end
    end

    local n = 0
    local function loop()
        enqueue()
        if not pool:hasjob() then
            return nil
        end
        pool:dojob()
        if pool:haserror() then
            pool:synchronize()
        end
        enqueue()
        n = n + 1
        return n, sample 
    end

    return loop
end

