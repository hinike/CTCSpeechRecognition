--Retrieves audio datasets. Currently retrieves the AN4 dataset by giving the folder directory.
require 'lfs'
require 'xlua'
require 'lmdb'
require 'torch'
require 'Mapper'

-- manipulate with this object
local util = {}

local function split(s, p)
    local rt = {}
    string.gsub(s, '[^' .. p .. ']+', function(w) table.insert(rt, w) end)
    return rt
end

local function trans2tokens(line, _mapper)
    --[[
        input:
            line: ERASE C Q Q F SEVEN (id)

        output:
            label: {3,7,1,2,8}
            line: erase c q q f seven
    --]]

    local label = {}
    line = string.lower(line)
    line = line:gsub('^%s', ''):gsub('', ''):gsub('', ''):gsub('%(.+%)', ''):gsub('%s$', ''):gsub('<s>', ''):gsub('</s>', '')
    -- strip
    line = line:match("^%s*(.-)%s*$")
    for i = 1, #line do
        local character = line:sub(i, i)
        table.insert(label, _mapper.alphabet2token[character])
    end

    return label, line
end

local function start_txn(_path, _name)
    local db = lmdb.env {
        Path = _path,
        Name = _name
    }
    db:open()
    local txn = db:txn()
    return db, txn
end

local function end_txn(db, txn)
    txn:commit()
    txn:abort()
    db:close()
end

function util.mk_lmdb(root_path, index_path, dict_path, out_dir, windowSize, stride)
    --[[
        read index and dict files and make lmdb

        input:
            index_path: path to the index file
            dict_path: path to the dict file
            out_dir: dir to store lmdb
            windowSize, stride: hyperparas for making spects

        NOTE:
            line sturct of dict file: <char/word>

            line struct of index file: <wave_file_path>@<transcript>@,
            where wave_file_path should be absolute path
    --]]

    require 'audio'
    local startTime = os.time()
    local mapper = Mapper(dict_path)

    -- start writing
    local db_spect, txn_spect = start_txn(out_dir .. '/spect', 'spect')
    local db_label, txn_label = start_txn(out_dir .. '/label', 'label')
    local db_trans, txn_trans = start_txn(out_dir .. '/trans', 'trans')

    local cnt = 1
    local saved_cnt = 1
    local show_gap = 100
    for line in io.lines(index_path) do
        -- print('processing ' .. line .. ' cnt: ' .. cnt)
        local wave_path, trans = split(line, '@')[1], split(line, '@')[2]

        -- make label
        local label, modified_trans = trans2tokens(trans, mapper)

        -- make spect
        local wave = audio.load(root_path .. wave_path)
        local spect = audio.spectrogram(wave, windowSize, 'hamming', stride) -- freq-by-frames tensor
        
        if spect:size(2) >= #label then -- ensure more frames than label
            -- put into lmdb
            txn_spect:put(cnt, spect:float())
            txn_label:put(cnt, torch.serialize(label))
            txn_trans:put(cnt, torch.serialize(modified_trans))
            saved_cnt = saved_cnt + 1
        end

        -- commit buffer
        if cnt % show_gap == 0 then
            txn_spect:commit(); txn_spect = db_spect:txn()
            txn_label:commit(); txn_label = db_label:txn()
            txn_trans:commit(); txn_trans = db_trans:txn()
        end

        xlua.progress(cnt % show_gap + 1, show_gap)
        cnt = cnt + 1
    end
    print('total '..cnt..' items scanned, '..saved_cnt..' items saved in ' .. os.time() - startTime .. 's')
    -- close
    end_txn(db_spect, txn_spect)
    end_txn(db_label, txn_label)
    end_txn(db_trans, txn_trans)
end

function util.get_lens(lmdb_path)
    --[[
        reads out train and test lmdb and put data lengthes into a file

        NOTE: only for spect features
    --]]
    
    
    local db_train = lmdb.env{Path = lmdb_path..'/train/spect/', Name='spect'}
    local db_test = lmdb.env{Path = lmdb_path..'/test/spect/', Name='spect'}
    
    db_train:open()
    db_test:open()

    local num_train = db_train:stat()['entries']
    local num_test = db_test:stat()['entries']

    local txn = db_train:txn(true)
    for i=1,num_train do
        local tensor = txn:get(i)

        local f = io.open('lmdb_lens', 'a')
        f:write(string.format('%i\n',tensor:size(2)))
        f:close()
    end
    txn:abort()
    db_train:close()
    
    local f = io.open('lmdb_lens', 'a')
    f:write('====================================')
    f:close()

    txn = db_test:txn(true)
    for i=1,num_test do
        local tensor = txn:get(i)

        local f = io.open('lmdb_lens', 'a')
        f:write(string.format('%i\n',tensor:size(2)))
        f:close()
    end
    txn:abort()
    db_test:close()

end

return util
