from src.utils import activation_fns


class Activation:
    def __init__(self, activation_fn):
        self.activation_fn = activation_fn

    def init_params(self, prev_units):
        self.units = prev_units
        self.params = None
        self.grads = None

    def forward(self, input, predict=False):
        output = self.activation_fn(input)
        assert(output.shape == input.shape)

        if not predict:
            self.cache = input
        return output

    def backward(self, dinput, Y):
        input = self.cache

        if self.activation_fn == activation_fns.softmax:
            doutput = activation_fns.softmax_delta(input, Y)
        else:
            doutput = dinput * self.activation_fn(input, delta=True)
        assert(doutput.shape == input.shape)

        self.cache = None
        return doutput
