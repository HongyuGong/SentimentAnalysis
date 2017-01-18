"""
classification with TensorFlow
"""

import tensorflow as tf
import os
import numpy as np
import data_util
from vocab import Vocab


def fnn(train_x, train_y, test_x, test_y):
    input_dim = len(train_x[0])
    output_dim  = len(train_y[0])
    print "input dim", input_dim, "output dim", output_dim

    # Define variables
    x = tf.placeholder(tf.float32, [None, input_dim])
    W = tf.Variable(tf.zeros([input_dim, output_dim]))
    b = tf.Variable(tf.zeros([output_dim]))
    y = tf.nn.softmax(tf.matmul(x, W)+b) # actual output y
    y_ = tf.placeholder(tf.float32, [None, output_dim]) # gold output y_

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    wrong_prediction = tf.not_equal(tf.argmax(y,1), tf.argmax(y_, 1))
    error_rate = tf.reduce_mean(tf.cast(wrong_prediction, tf.float32))

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # Initialization
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    # Train the model
    print "training ..."
    iter_num = 1600
    for i in range(iter_num):
        sess.run(train_step, feed_dict={x: train_x, y_: train_y})

    train_acc = sess.run(accuracy, feed_dict={x: train_x, y_: train_y})
    print "training accuracy:", train_acc
                
    # Test
    print "testing ..."
    test_acc = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
    print "test accuracy:", test_acc # accuracy

def getAvgVecsFromIdx(idx_list):
    vecDim = 300
    directory = "../embedding/"
    vocabInputFile = "vocab.txt"
    vectorInputFile = "vectors.bin"
    vocab = Vocab(vecDim, directory, vocabInputFile, vectorInputFile)

    vec_avg_list = []
    vec_list = vocab.getVecFromId(idx_list)
    for vec in vec_list:
        vec_sum = np.sum(vec, axis=0)
        vec_avg = vec_sum / len(vec)
        vec_avg_list.append(vec_avg)
    print "vec shape:", vec.shape
    print "vec_avg dim:", len(vec_avg),"should be", 300
    return np.array(vec_avg_list)


if __name__=="__main__":

    """stanford sentiment treebank"""
    # binary classification
    train_features_idx, train_binary_labels, test_features_idx, test_binary_labels = data_util.loadStanfordDataset(2, True)
    train_features = getAvgVecsFromIdx(train_features_idx)
    test_features = getAvgVecsFromIdx(test_features_idx)
    fnn(train_features, train_binary_labels, test_features, test_binary_labels)

    # five-class classification
    train_features_idx, train_multi_labels, test_features_idx, test_multi_labels = data_util.loadStanfordDataset(5, True)
    train_features = getAvgVecsFromIdx(train_features_idx)
    test_features = getAvgVecsFromIdx(test_features_idx)
    fnn(train_features, train_multi_labels, test_features, test_multi_labels)
    

    """IMDB review dataset"""
    # binary classification
    train_features_idx, train_binary_labels, test_features_idx, test_binary_labels = data_util.loadIMDBDataset(2, True)
    train_features = getAvgVecsFromIdx(train_features_idx)
    test_features = getAvgVecsFromIdx(test_features_idx)
    fnn(train_features, train_binary_labels, test_features, test_binary_labels)

    # five-class classification
    train_features_idx, train_binary_labels, test_features_idx, test_binary_labels = data_util.loadIMDBDataset(5, True)
    train_features = getAvgVecsFromIdx(train_features_idx)
    test_features = getAvgVecsFromIdx(test_features_idx)
    fnn(train_features, train_binary_labels, test_features, test_binary_labels)



