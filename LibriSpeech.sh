th Train.lua \
  -trainingSetLMDBPath   /data1/nfs_share/data/LibriSpeech/320_160/train \
  -validationSetLMDBPath /data1/nfs_share/data/LibriSpeech/320_160/test \
  -nGPU 4 \
  -dataHeight 161 \
  -hidden_size 600 \
  -num_layers 7 \
  -rnn_type GRU \
  -batchSize 100 \
  -validationBatchSize 24 \
  -learning_rate 1e-3 \
  -saveModelIterations 20 \
  -dictionaryPath ./dictionary \
  -dictSize 29 \
  -learning_rate_decay_every 20 \
  2>&1 | tee $(date +%Y%m%d-%T).log
