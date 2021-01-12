import numpy as np
import mnist
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
import numpy as np
import time


class NeuralNetwork():
  def __init__(self, hparams, x_train, y_train, x_test, y_test, network_params):
    self.hparams = hparams
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.network_params = network_params

  def prep_labels(train_labels):
    # TODO - define y_star : one hot encoded ground truth labels
    y_star = np.zeros((len(train_labels), 10))
    for label in train_labels:
      for i in range(10):
        if train_labels[label] == i:
          y_star[label][i] = 1
    return y_star

  # Forward
  def sigmoid(self, x):
      return 1. / (1 + np.exp(-x))

  def stable_softmax(self, X):
      exps = np.exp(X - np.max(X))
      return exps / np.sum(exps)


  # Backward
  def d_sigmoid(self, x):
      act = self.sigmoid(x)
      return act * (1 - act)

  def d_softmax(self, x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))

  # Init parameters using a normal dist : (0, 0.05)
  def init_params(num_hidden=160, input_size=784, output_size=10):

      # initialize parameters
      W1 = np.random.normal(0, 0.05, (num_hidden, input_size))
      W2 = np.random.normal(0, 0.05, (output_size, num_hidden))
      B1 = np.random.normal(0, 0.05, (num_hidden))
      B2 = np.random.normal(0, 0.05, (output_size))

      return {"W1": W1, "W2" : W2, "B1": B1, "B2": B2}

  def forward_pass(self, x_flattened):
    params = self.network_params
    # input layer activations becomes sample
    params['A0'] = x_flattened

    # input layer to hidden layer 1
    params['Z1'] = np.dot(params["W1"], params['A0'])
    params['A1'] = self.sigmoid(params['Z1'])

    # hidden layer 1 to output
    params['Z2'] = np.dot(params["W2"], params['A1'])
    params['A2'] = self.stable_softmax(params['Z2'])

    return params['A2']

  def backprop(self, y, output):
      backpropagate = {}
      # Calculate W1 update
      error = 2 * (output - y) / output.shape[0] * self.d_softmax(self.network_params["Z2"])
      backpropagate["W2"] = np.outer(error, self.network_params["A1"])

      # Calculate W0 update
      error = np.dot(self.network_params["W2"].T, error) * self.d_sigmoid(self.network_params["Z1"])
      backpropagate["W1"] = np.outer(error, self.network_params["A0"])

      return backpropagate

  def train(self, x_train, y_train, x_test, y_test, max_ite=100):
    # Start : Convert the training images to a vector images.
    #self.train_flat = np.array([image.reshape(-1, 1) for image in x_train])
    time1 = time.time()
    for iteration in range(max_ite):
      for x,y in zip(x_train, y_train):
          output = self.forward_pass(x.reshape(-1, 1))
          loss = 0.5 * np.linalg.norm(y - output)
          backprop = self.backprop(np.reshape(y, (y.shape[0], 1)), output)
          network_params = self.update_network(backprop)
          time2 = time.time()
          accuracy = self.check_accuracy(x_test, y_test)
          print("Accuracy:{0}, iteration: {1}, time: {2}, loss: {3}".format(accuracy, iteration, time2 - time1, loss))
    return accuracy

  def update_network(self, backprop):
    network_params = {}
    for key, value in backprop.items():
      self.network_params[key] -= self.hparams["l_rate"] * value

    return network_params


  def check_accuracy(self, x_test, y_test):
    '''
    counter = 0
    S = y_test.shape[0]
    for x_, y_ in zip(x_test, y_test):
      output = self.forward_pass(x_.reshape(-1, 1))
      pred = np.argmax(output)
      y_val = np.argmax(y_)
      if pred == y_val:
        counter = counter + 1

    accu = counter / S
    return accu
    '''
    predictions = []

    for x, y in zip(x_test, y_test):
        output = self.forward_pass(x.reshape(-1, 1))
        pred = np.argmax(output)
        predictions.append(pred == np.argmax(y))

    return np.mean(predictions)


  def total_loss(loss_list):
    size = loss_list.size
    return np.sum(loss_list)/size

if __name__ == "__main__":
    #load the data
    train_data_size = 6000 # Only a subset of training data is utilized, dont change for submission
    test_data_size = 1000
    train_data = mnist.load_data("train-images-idx3-ubyte.gz")[0:train_data_size]
    train_labels = mnist.load_labels("train-labels-idx1-ubyte.gz")[0:train_data_size]
    test_data = mnist.load_data("t10k-images-idx3-ubyte.gz")[train_data_size:train_data_size + test_data_size]
    test_labels = mnist.load_labels("t10k-labels-idx1-ubyte.gz")[train_data_size:train_data_size + test_data_size]

    y_train = NeuralNetwork.prep_labels(train_labels)
    y_test = NeuralNetwork.prep_labels(test_labels)
    # Initialise the Neural Network
    hparams = {"l_rate": 0.001}
    np.random.seed(42)

    network_params = NeuralNetwork.init_params(num_hidden=160, input_size=784, output_size=10)
    feedforwardnn = NeuralNetwork(hparams, train_data, y_train, test_data, y_test, network_params)
    # Train
    feedforwardnn.train(train_data, y_train, test_data, y_test, 10)
    # Evaluate

    # print progress
    '''
    if it % 50 == 0:
        #print ('it=', it, 'loss=', L[it])
        plt.figure(101, figsize=(8,4))
        plt.clf()
        fig = plt.gcf()
        plt.subplot(121)
        #y_plot = np.asarray(y_plot).T
        plt.plot(y_star[0,:], y_star[1,:], 'rx', label='ground truth')
        #plt.plot(y_plot[0,:], y_plot[1,:], 'gx', label='predictions')
        plt.legend(loc='best')
        plt.title('it = ' + str(it))
        fig.canvas.draw()

        plt.subplot(122)
        plt.plot(loss_list[:it], label='Loss')
        plt.grid(True)
        plt.yscale('log')
        plt.legend(loc='best')
        fig.canvas.draw()

        time.sleep(0.01)
      '''





