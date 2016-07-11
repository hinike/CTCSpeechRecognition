git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
source ~/.bashrc
source ~/.zshrc
source ~/.profile
sudo apt-get install libfftw3-dev sox libsox-dev libsox-fmt-all
luarocks install cutorch
luarocks install cunn
luarocks install cunnx
luarocks install http://raw.githubusercontent.com/baidu-research/warp-ctc/master/torch_binding/rocks/warp-ctc-scm-1.rockspec
luarocks install optim
luarocks install rnn
luarocks install nnx
luarocks install xlua
luarocks install nngraph
luarocks install tds
luarocks install lmdb.torch
luarocks install https://raw.githubusercontent.com/soumith/lua---audio/master/audio-0.1-0.rockspec
luarocks install cudnn