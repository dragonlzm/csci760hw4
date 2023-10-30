import numpy as np


class my_linear:
    # can be extend to using a for loop to connect the backprop of each layer
    def __init__(self, input_dim, output_dim, init_val=None):
        # given the input dimension and the output dimension of the linear layer 
        # and the initialization method of the layer
        # create the param variable and initial the layer
        if init_val is None:
            self.params_w = np.random.normal(0, 0.1, [input_dim, output_dim])
        else:
            self.params_w = np.full((input_dim, output_dim), init_val)
        # create the gradient variable and initial
        self.gradient_w = np.zeros([input_dim, output_dim])

    def forward(self, X):
        # X: NxD numpy array, each row is an input image
        # f_output: NxD numpy array, each 'row' is an output of a image after linear layer
        # do the forward pass
        f_output = np.matmul(X, self.params_w)
        # save the forward value
        self.forward_value = f_output
        return f_output

    def backward(self, X, ds_grad):
        # X: NxD numpy array, the input of the forward pass.
        # ds_grad: NxD numpy array, each row is the partial derivative of the mini-batch loss 
        # with respect ti f_output[i] from down-stream layer.
        # b_output: NxD numpy array, each row is the partial derivatives of the mini-batch loss 
        # with respect to the forward input X[i].

        # calculate the gradients
        self.gradient_w = np.matmul(X[:, :, np.newaxis], ds_grad[:, np.newaxis, :]).sum(axis=0)

        # the gradient that will be sent to the up-stream layer for grandient calculateion
        b_output = np.matmul(self.params_w, ds_grad.transpose()).transpose()
        return b_output


class my_sigmoid:
    def forward(self, X):
        # the forward pass of the sigmoid
        f_output = 1 / (1 + np.exp(-X))
        self.f_value = f_output
        return f_output

    def backward(self, X, ds_grad):
        # the backward pass of the sigmoid
        sigmoid_val = 1 / (1 + np.exp(-X))
        backward_output = sigmoid_val * (1-sigmoid_val) * ds_grad
        return backward_output
    