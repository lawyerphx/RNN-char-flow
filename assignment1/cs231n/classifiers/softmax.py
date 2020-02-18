from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    mult = X.dot(W)
    pred = np.exp(mult - np.max(mult, axis=1).reshape(-1,1))
    dpred = np.zeros(pred.shape)
    for i in range(num_train):
        pred[i] /= np.sum(pred[i])
        loss -= np.log(pred[i][y[i]])
        for j in range(num_classes):
            dpred[i][j] = pred[i][j]
        dpred[i][y[i]] = pred[i][y[i]]-1
    loss = loss/num_train + reg * np.sum(W * W)
    dW = np.dot(X.T, dpred)/num_train + 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    mult = X.dot(W)
    pred = np.exp(mult - np.max(mult, axis=1).reshape(-1,1))
    pred = pred / np.sum(pred, axis=1).reshape(-1,1)
    loss = np.sum(-np.log(pred[range(num_train), list(y)]))/num_train + reg * np.sum(W * W)

    dpred = pred.copy()
    dpred[range(num_train), list(y)] -= 1
    dW = np.dot(X.T, dpred)/num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
