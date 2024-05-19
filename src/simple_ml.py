import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, 'r') as f:
        _ = int.from_bytes(f.read(4), 'big')
        num_items =  int.from_bytes(f.read(4), 'big')
        num_row = int.from_bytes(f.read(4), 'big')
        num_col = int.from_bytes(f.read(4), 'big')

        image_buffer = f.read(num_items * num_row * num_col)
        imgs = np.frombuffer(image_buffer, dtype=np.uint8).astype(np.float32).reshape(num_items, num_row * num_col)
        imgs /= 255.0
        # for debugging
        # visualizer_helper(imgs[0])

    with gzip.open(label_filename, 'r') as f:
        _ = int.from_bytes(f.read(4), 'big')
        num_items =  int.from_bytes(f.read(4), 'big')
        label_buffer = f.read(num_items)
        labels = np.frombuffer(label_buffer, dtype=np.uint8)
    return (imgs, labels)
    ### END YOUR CODE


def visualizer_helper(img):
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap='gray')
    plt.show()


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    log_sum_exp = np.log(np.sum(np.exp(Z), axis=1))
    loss = -Z[np.arange(y.shape[0]), y] + log_sum_exp
    ### END YOUR CODE
    return np.mean(loss)


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    def update_gradient(X, y, theta, lr, batch):
        logit = np.exp(np.matmul(X,theta))
        z = logit / np.sum(logit, axis=1, keepdims=True)
        grad = np.matmul(X.T, (z - OneHot(y, theta.shape[1])))
        theta -= lr * grad / batch
    iter = X.shape[0] // batch
    for i in range(iter):
        Xi, yi = X[batch*i:batch*(i+1)], y[batch*i:batch*(i+1)]
        update_gradient(Xi, yi, theta, lr, batch)
    update_gradient(X[batch*iter:], y[batch*iter:], theta, lr, batch)

    ### END YOUR CODE

def OneHot(x, num_label=10):
    one_hot = np.zeros((x.shape[0], num_label), dtype=np.int8)
    np.put_along_axis(one_hot, np.expand_dims(x, axis=1), 1, axis=1)
    return one_hot


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    def update_gradient(X, y, W1, W2, lr, batch):
        Z1 = ReLU(np.matmul(X, W1)) # num_examples X hidden_dim
        G2 = Softmax(np.matmul(Z1, W2)) - OneHot(y, W2.shape[1]) 
        G1 = np.where(Z1>0, 1, 0)*(np.matmul(G2,W2.T))
        W1_grad = 1/batch * np.matmul(X.T, G1)
        W2_grad = 1/batch * np.matmul(Z1.T, G2)
        W1 -= lr * W1_grad
        W2 -= lr * W2_grad

    iter = X.shape[0] // batch
    for i in range(iter):
        Xi, yi = X[batch*i:batch*(i+1)], y[batch*i:batch*(i+1)]
        update_gradient(Xi, yi, W1, W2, lr, batch)
    update_gradient(X[batch*iter:], y[batch*iter:], W1, W2, lr, batch)


    ### END YOUR CODE

def ReLU(x):
    return np.where(x>=0, x, 0)

def Softmax(X):
    ex = np.exp(X)
    return ex / np.sum(ex, axis=1, keepdims=True)

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)

