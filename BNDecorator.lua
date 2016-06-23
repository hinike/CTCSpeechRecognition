require 'dpnn'

local BNDecorator, parent = torch.class("nn.BNDecorator", "nn.Decorator")

function BNDecorator:__init(module)
    parent.__init(self, module)
    assert(torch.isTypeOf(module, 'nn.Module'))
end

function BNDecorator:updateOutput(input)
    local T, N = input:size(1), input:size(2)
    self._input = input:view(T * N, -1)
    self.output = self.module:updateOutput(self._input)
    self.output = self.output:view(T, N, -1)
    return self.output
end

function BNDecorator:updateGradInput(input, gradOutput)
    local T, N = input:size(1), input:size(2)
    self._gradOutput = gradOutput:view(T * N, -1)
    self.gradInput = self.module:updateGradInput(self._input, self._gradOutput)
    self.gradInput = self.gradInput:view(T, N, -1)
    return self.gradInput
end

function BNDecorator:accGradParameters(input, gradOutput, scale)
    self.module:accGradParameters(self._input, self._gradOutput, scale)
end

function BNDecorator:accUpdateGradParameters(input, gradOutput, lr)
    self.module:accUpdateGradParameters(self._input, self._gradOutput, lr)
end

function BNDecorator:sharedAccUpdateGradParameters(input, gradOutput, lr)
    self.module:sharedAccUpdateGradParameters(self._input, self._gradOutput, lr)
end
