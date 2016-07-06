local BRNN, parent = torch.class('cudnn.BRNN', 'cudnn.RNN')

function BRNN:__init(inputSize, hiddenSize, numLayers, batchFirst)
    parent.__init(self,inputSize, hiddenSize, numLayers, batchFirst)
    self.bidirectional = 'CUDNN_BIDIRECTIONAL'
    self.mode = 'CUDNN_RNN_RELU'
    self.numDirections = 2
    self:reset()
end
