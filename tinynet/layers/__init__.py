from abc import ABC, abstractmethod

class Layer(ABC):
    """Base initializer class"""

    def __init__(self, *args, **kwargs):
        # Should be defined in init_params

        # Required for configuring the whole model
        self.in_shape = None
        self.out_shape = None
        
        # Required for updating the layer's parameters
        self.params = None
        self.grads = None

        # Should be defined in forward
        self.cache = None

    
    @abstractmethod
    def init_params(self, in_shape, *args, **kwargs):
        """Initialize the layer using hyperparameters."""
        pass


    @abstractmethod
    def forward(self, X, predict=False, *args, **kwargs):
        """Do a forward pass through the layer.
        
        `X` is the output tensor from the previous layer.
        """
        pass


    @abstractmethod
    def backward(self, dout, *args, **kwargs):
        """Do a backward pass through the layer.
        
        `dout` is the tensor derivative of the input X from the next layer.
        """
        pass