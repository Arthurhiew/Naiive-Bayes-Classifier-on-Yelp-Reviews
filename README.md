# Naiive-Bayes-Classifier-on-Yelp-Reviews
A Naiive Bayes Classifier that takes in Yelp reviews as input to determine whether the reviews are positive or negative.

## Pre-processing step
This step converts each sentence in into a feature vector (one-hot encoding) and each sentence are represented as a "bag of words".
1. strip the punctuation
2. convert train and test set into one hot encoding
3. output the preprocessed data into two saperate files

## To Run
In the terminal do `python bayes.py`

Training accuracy: 0.9679
Test accuracy: 0.7746
