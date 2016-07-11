th AN4.lua \
  -trainingSetLMDBPath ./prepare_an4/train/ \
  -validationSetLMDBPath ./prepare_an4/test/ \
  -dataHeight 161 \
  -hidden_size 1000 \
  -num_layers 7 \
  -rnn_type rnn \
  -batchSize 100 \
  -learning_rate 1e-3 \
  -saveModelIterations 70 \
  -dictSize 28 \
  2>&1 | tee log_$(date +%Y%m%d-%T).txt
