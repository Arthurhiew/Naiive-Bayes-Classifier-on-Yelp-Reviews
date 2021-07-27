import numpy as np
import pandas as pd
import string
import logging

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


def load_data():
    # read file into np array from pandas
    train = np.array(pd.read_csv("trainingSet.txt", delimiter="\t", names=["review", "classlabel"]))
    test = np.array(pd.read_csv("testSet.txt", delimiter="\t", names=["review", "classlabel"]))
    # separate dataset into x and y
    x_train_pre, y_train = train.T
    x_test_pre, y_test = test.T
    # remove punctuation
    x_train = [
        np.array(str(line).translate(str.maketrans("", "", string.punctuation)).lower().split()) for line in x_train_pre
    ]

    # test = pd.DataFrame(x_train)
    # print(test)

    x_test = [
        np.array(str(line).translate(str.maketrans("", "", string.punctuation)).lower().split()) for line in x_test_pre
    ]
    # convert it back to np array
    x_train = np.array(x_train, dtype=object)
    x_test = np.array(x_test, dtype=object)
    # get header
    header = np.unique(np.array(str(x_train_pre).translate(str.maketrans("", "", string.punctuation)).lower().split()))
    return x_train, y_train, x_test, y_test, header


# convert x_train and x_test into one hot encoding
def convert_to_one_hot(classes, X):
    one_hot = np.zeros((len(X), classes.shape[0]), dtype="int8")
    for row in range(len(X)):
        for head in X[row]:
            one_hot[row][np.where(head == classes)] = 1
    return one_hot


def write_data_to_file(filename, header, dataset, classlabel):
    with open(filename, "w") as f:
        print(*header, "classlabel", sep=",", file=f)
        for data in dataset:
            print(*data, sep=",", file=f)
    f.close()


class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, X, Y, header):
        self.X = X
        self.Y = Y
        self.header = header
        self.total_pos = X[Y == 1].shape[0]
        self.total_neg = X[Y == 0].shape[0]

    # prior: the probability of word given class P(w|c)
    # row1:[p(w0|y0),p(w1|y0)...p(wn|y0)],
    # row2:[p(w0|y1),p(w1|y1)...p(wn|y1)],
    def dirichlet_prior(self):
        # number of positive and negative sample
        positive_sample = self.X[self.Y == 1]
        negative_sample = self.X[self.Y == 0]
        # dirichletr prior #class = 2
        pos_likelihood = (np.sum(positive_sample, axis=0) + 1) / (positive_sample + 2)
        neg_likelihood = (np.sum(negative_sample, axis=0) + 1) / (negative_sample + 2)
        return pos_likelihood, neg_likelihood

    def posterior(self, x):
        prob_pos = self.total_pos / (self.total_pos + self.total_neg)
        prob_neg = 1 - prob_pos
        pos_likelihood, neg_likelihood = self.dirichlet_prior()
        pos_posterior = np.log(self.total_pos) + np.sum(x * np.log(pos_likelihood))
        neg_posterior = np.log(self.total_neg) + np.sum(x * np.log(neg_likelihood))

        return pos_posterior, neg_posterior

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            pos_posterior, neg_posterior = self.posterior(X[i])
            if pos_posterior > neg_posterior:
                y_pred[i] = 1
        return y_pred

    def evaluate(self, y_pred, y_test):
        return np.mean(y_pred == y_test)


if __name__ == "__main__":
    logging.info("Loading data")
    x_train, y_train, x_test, y_test, header = load_data()

    logging.info("\n---------------------------------------------------------------------------\n")

    logging.info("Converting data set to one-hot representation")
    x_train = convert_to_one_hot(header, x_train)
    x_test = convert_to_one_hot(header, x_test)

    logging.info("\n---------------------------------------------------------------------------\n")

    logging.info("Writing to file")
    write_data_to_file("preprocessed_train.txt", header, x_train, y_train)
    write_data_to_file("preprocessed_test.txt", header, x_test, y_test)

    logging.info("\n---------------------------------------------------------------------------\n")
    logging.info("Training Bayes")
    clf = NaiveBayes()
    clf.fit(x_train, y_train, header)
    # train set

    logging.info("Making prediction (Training set)")
    y_pred = clf.predict(x_train)
    train_accuracy = clf.evaluate(y_pred, y_train)
    logging.info("Train accuracy: {:.4}".format(train_accuracy))

    logging.info("\n---------------------------------------------------------------------------\n")

    logging.info("Making prediction (Test set)")
    y_pred = clf.predict(x_test)
    test_accuracy = clf.evaluate(y_pred, y_test)
    logging.info("Test accuracy: {:.4}".format(test_accuracy))

    logging.info("\n---------------------------------------------------------------------------\n")
    logging.info("Writing to result.txt")

    with open("result.txt", "w") as f:
        print("trainingSet.txt", file=f)
        print("Training accuracy: {:.4}".format(train_accuracy), file=f)
        print("testSet.txt", file=f)
        print("Test accuracy: {:.4}".format(test_accuracy), file=f)
    logging.info("Done.")
    logging.info("\n---------------------------------------------------------------------------\n")