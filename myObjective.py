import theano
import theano.tensor as T
from theano.tensor.nnet import binary_crossentropy, categorical_crossentropy
from lasagne.regularization import l2
import lasagne
def mse(x, t):
    """Calculates the MSE mean across all dimensions, i.e. feature
     dimension AND minibatch dimension.

    :parameters:
        - x : predicted values
        - t : target values

    :returns:
        - output : the mean square error across all dimensions
    """
    return (x - t) ** 2

class Objective(object):
    _valid_aggregation = {None, 'mean', 'sum'}

    """
    Training objective

    The  `get_loss` method returns cost expression useful for training or
    evaluating a network.
    """
    def __init__(self, input_layer, loss_function=mse, l2_strength = 0, aggregation='mean'):
        """
        Constructor

        :parameters:
            - input_layer : a `Layer` whose output is the networks prediction
                given its input
            - loss_function : a loss function of the form `f(x, t)` that
                returns a scalar loss given tensors that represent the
                predicted and true values as arguments..
            - aggregation : either:
                - `'mean'` or `None` : the mean of the the elements of the
                loss will be returned
                - `'sum'` : the sum of the the elements of the loss will be
                returned
        """
        self.input_layer = input_layer
        self.loss_function = loss_function
        self.target_var = T.matrix("target")
        self.l2_strength = l2_strength
        if aggregation not in self._valid_aggregation:
            raise ValueError('aggregation must be \'mean\', \'sum\', '
                             'or None, not {0}'.format(aggregation))
        self.aggregation = aggregation

    def get_loss(self, input=None, target=None, aggregation=None, **kwargs):
        """
        Get loss scalar expression

        :parameters:
            - input : (default `None`) an expression that results in the
                input data that is passed to the network
            - target : (default `None`) an expression that results in the
                desired output that the network is being trained to generate
                given the input
            - aggregation : None to use the value passed to the
                constructor or a value to override it
            - kwargs : additional keyword arguments passed to `input_layer`'s
                `get_output` method

        :returns:
            - output : loss expressions
        """
        network_output = lasagne.layers.get_output(self.input_layer, input, **kwargs)
        if target is None:
            target = self.target_var
        if aggregation not in self._valid_aggregation:
            raise ValueError('aggregation must be \'mean\', \'sum\', '
                             'or None, not {0}'.format(aggregation))
        if aggregation is None:
            aggregation = self.aggregation

        losses = self.loss_function(network_output, target) + self.l2_strength*l2(self.input_layer)

        if aggregation is None or aggregation == 'mean':
            return losses.mean()
        elif aggregation == 'sum':
            return losses.sum()
        else:
            raise RuntimeError('This should have been caught earlier')
