#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a 
        y_hat_i = self.predict(x_i)
        if y_hat_i != y_i:
            self.W[y_i] += x_i
            self.W[y_hat_i] -= x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        ###### Wrong. Why? #####
        # Q1.1b
        prediction = self.predict(x_i)
        y_hat_i = 1 / (1 + np.exp(-prediction))
        error = y_i - y_hat_i
        self.W += learning_rate * error * x_i.T

class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.b1 = np.zeros((hidden_size, 1))
        print(self.b1)
        self.b2 = np.zeros((n_classes, 1))
        w1 = []
        for i in range (hidden_size):
            w1_row = []
            for j in range(n_features):
                w1_row.append(np.random.normal(0.1, 0.01))
            w1.append(w1_row)
        self.W1 = np.array(w1)
        w2 = []
        for i in range(n_classes):
            w2_row = []
            for j in range(hidden_size):
                w2_row.append(np.random.normal(0.1, 0.01))
            w2.append(w2_row)
        self.W2 = np.array(w2)

    def predict(self, X):
        z1 = np.dot(self.W1, X.T) + self.b1
        x1 = []
        for i in range(z1.shape[0]):
            x1_row = []
            for j in range(z1.shape[1]):
                x1_row.append(z1[i][j] if z1[i][j] > 0 else 0)
            x1.append(x1_row)
        x1 = np.array(x1)
        z2 = np.dot(self.W2, x1.T) + self.b2
        sum_exp = np.sum(np.exp(z2))
        x2 = np.exp(z2) / sum_exp
        return x2.argmax(axis=0)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate() 
        # FIXME : should this use loss or same as linear classifier?
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        """
        Dont forget to return the loss of the epoch.
        """
        z1 = np.dot(self.W1, X.T) + self.b1
        x1 = np.array(list(map(lambda x: list(map(lambda k: k if k > 0 else 0, x)), z1)))
        print()
        z2 = np.dot(self.W2, x1) + self.b2
        sum_exp = np.sum(np.exp(z2))
        x2 = np.exp(z2) / sum_exp
        delta2 = x2 - y
        derivative1 = np.array(list(map(lambda x: list(map(lambda k: 1 if k > 0 else 0, x)), z1)))
        delta1 = np.dot(self.W2.T, delta2) * derivative1
        self.W2 -= learning_rate * np.dot(delta2, x1.T)
        self.b2 -= learning_rate * delta2
        self.W1 -= learning_rate * np.dot(delta1, X)
        self.b1 -= learning_rate * delta1        
        return -np.sum(y * np.log(x2))

def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()
