import numpy as np
import json
from copy import copy

def myminiBatchGradientDescent(model, lr):
    for module_name, layer in model.items():
        #print(module_name, hasattr(layer, 'params_w'))
        # if a module has learnable parameters
        if hasattr(layer, 'params_w'):
            #print("updating")
            # get the grandients
            g = layer.gradient_w
            # update a step of SGD
            layer.params_w -= g * lr
    return model


class softmax_and_cross_entropy_loss:
    def __init__(self):
        # expanded label
        self.expanded_Y = None
        # save the prediction
        self.pred_prob = None

    def forward(self, X, Y):
        # expand the label y to a one-hot vector
        # find the index 
        bs, cls_num = X.shape[0], X.shape[1]
        gt_idx = Y.astype(int).reshape(-1) + np.arange(bs) * cls_num
        # assign the value
        self.expanded_Y = np.zeros(X.shape).reshape(-1)
        self.expanded_Y[gt_idx] = 1.0
        self.expanded_Y = self.expanded_Y.reshape(X.shape)

        # to prevent the exp explode
        calibrated_X = X - np.amax(X, axis=1, keepdims=True)
        # calculate the softmax score
        sum_of_logit = np.sum(np.exp(calibrated_X), axis=1, keepdims=True)
        self.pred_prob = np.exp(calibrated_X) / sum_of_logit

        # calculate the loss
        loss_on_gt = np.multiply(self.expanded_Y, np.log(self.pred_prob))
        f_output = - np.sum(loss_on_gt)
        return f_output

    def backward(self, X, Y):
        # calculate the backprop grad
        b_output = - (self.expanded_Y - self.pred_prob) / X.shape[0]
        return b_output


class MyDataloader:
    # create a dataloader
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.size, self.data_dim = self.X.shape
        
    def get_example(self, idx):
        idx = np.array(idx)
        batchX = np.zeros((len(idx), self.data_dim))
        batchY = np.zeros((len(idx), ))
        batchX = self.X[idx]
        batchY = self.Y[idx]
        batchY = np.expand_dims(batchY, axis=-1)
        return batchX, batchY
