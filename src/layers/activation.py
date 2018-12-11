from src import activation_fns


class Activation:
    """
    Activation layer
    """

    def __init__(self, activation_fn):
        self.activation_fn = activation_fn

    def init_params(self, shape_in):
        self.shape_in = shape_in
        self.shape_out = shape_in

        self.params = None
        self.grads = None

    def forward(self, X, predict=False):
        out = self.activation_fn(X)

        if not predict:
            self.cache = X
        return out

    def backward(self, dout, Y):
        X = self.cache

        if self.activation_fn == activation_fns.softmax:
            dX = activation_fns.softmax_delta(X, Y)
            assert(dX.shape == X.shape)
        else:
            dX = dout * self.activation_fn(X, delta=True)
            assert(dX.shape == X.shape)

        self.cache = None
        return dX
