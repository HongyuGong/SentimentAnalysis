"""
one-layer neural net from scratch 
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import data_util
from vocab import Vocab
import numpy as np       
    
def train_model(classifier_type, features, labels, iter_num, epsilon, isPlot):
    # parameter initialization
    input_dim = len(features[0])
    output_dim = len(labels[0])
    np.random.seed(1563)
    W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
    b = np.zeros((1, output_dim))
    X = features
    model = {'weights': W, 'bias': b}
    err_list = []

    # gradient descent
    for i in range(iter_num):
        # print "W:", W, "b:", b
        
        W = model['weights']
        b = model['bias']
        #print "W:", W
        
        # update model parameters
        if (classifier_type == 'linear'):
            y = X.dot(W) + b
            W = W + epsilon * 2 * X.transpose().dot(labels-y)
            b = b + epsilon * 2 * np.sum(labels-y)

        elif (classifier_type == 'logistic'):
            y = 1/(1+np.exp(-(X.dot(W)+b)))
            W = W + epsilon * 2 * X.transpose().dot((labels-y) * (y-y*y))
            b = b + epsilon * 2 * np.sum((labels-y) * (y-y*y))

        elif (classifier_type == 'perceptron'):
            y = X.dot(W) + b
            W =  W + epsilon * X.transpose().dot(labels * (1*(y * labels <= 0)))
            b = b + epsilon * np.sum(labels * (1*(y * labels <= 0)))

        else: # linear SVM
	    C = 1
            y = X.dot(W) + b
            W = W + epsilon * (2 * W + C * X.transpose().dot(labels * (1*(y * labels <= 1))))
            b = b + epsilon * C * np.sum(labels * (1*(y * labels <= 1)))

        model['weights'] = W
        model['bias'] = b
        err = test(classifier_type, model, features, labels)
        err_list.append(err)

    # training error
    print "training error:", err_list[-1]
    # plot training error
    if (isPlot):
        plt.plot(range(iter_num), err_list)
        plt.xlabel('iterations')
        plt.ylabel('error rate')
        plt.title(classifier_type+' classifier')
        plt.axis([0,iter_num,0,1.0])
        plt.grid(True)
        plt.savefig(classifier_type+'.eps')
        plt.close()

    return model


def predict(classifier_type, model, features):
    W = model['weights']
    b = model['bias']
    vals = features.dot(W) + b
    pred_vals = []
    pred_labels = []

    # linear classifier
    if (classifier_type  == 'linear'):
        pred_vals = vals
        #print "pred_vals", pred_vals
        pred_labels = 1 * (pred_vals > 0.5)

    elif (classifier_type == 'logistic'):
        pred_vals = 1/(1+np.exp(-vals))
        pred_labels = 1 * (pred_vals > 0.5)

    elif (classifier_type == 'perceptron'):
        pred_vals = vals
        pred_labels = 2 * (pred_vals > 0) - 1

    else: # linear SVM
        pred_vals = vals
        pred_labels = 2 * (pred_vals > 0) - 1

    return pred_vals, pred_labels


def test(classifier_type, model, features, labels):
    pred_vals, pred_labels = predict(classifier_type, model, features)
    # MSE
    #mse = np.sum(np.square(pred_vals - labels))
    # accuracy
    acc = np.mean(np.equal(pred_labels, labels))

    #print "mse:", mse,  "acc:", acc
    #print "error rate:", err
    return acc

def runClassifiers(train_features, train_labels, test_features, test_labels):
    """
    plot convergence of each classifier
    """
    iter_num = 2000
    epsilon = 1e-5
    #model_dict = dict()
    isPlot = False

    # linear classifier
    print "linear classifier ..."
    classifier_type = 'linear'
    model = train_model(classifier_type, train_features, train_labels, iter_num, epsilon, isPlot)
    print "test error:", test(classifier_type, model, test_features, test_labels)
    #model_dict[classifier_type] = model.copy()

    # logistic classifier
    print "logistic classifier ..."
    classifier_type = 'logistic'
    model = train_model(classifier_type, train_features, train_labels, iter_num, epsilon, isPlot)
    print "test error:", test(classifier_type, model, test_features, test_labels)
    #model_dict[classifier_type] = model.copy()

    # perceptron
    print "perceptron ..."
    train_labels_perceptron = 2 * train_labels[:] - 1
    test_labels_perceptron = 2 * test_labels[:] - 1
    classifier_type = 'perceptron'
    model = train_model(classifier_type, train_features, train_labels_perceptron, iter_num, epsilon, isPlot)
    print "test error:",test(classifier_type, model, test_features, test_labels_perceptron)
    #model_dict[classifier_type] = model.copy()

    # linear SVM
    print "linear SVM ..."
    train_labels_svm = 2 * train_labels - 1
    test_labels_svm = 2 * test_labels - 1
    classifier_type = 'svm'
    model = train_model(classifier_type, train_features, train_labels_svm, iter_num, epsilon, isPlot)
    print "test accuracy:", test(classifier_type, model, test_features, test_labels_svm)
    #model_dict[classifier_type] = model.copy()

    #return model_dict

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
    """
    # load stanford dataset
    train_features_idx, train_labels, test_features_idx, test_labels = data_util.loadStanfordDataset(2, False)
    train_features = getAvgVecsFromIdx(train_features_idx)
    test_features = getAvgVecsFromIdx(test_features_idx)
    # run four classifiers
    runClassifiers(train_features, train_labels, test_features, test_labels)
    """

    # load IMDB dataset
    train_features_idx, train_labels, test_features_idx, test_labels = data_util.loadIMDBDataset(2, False)
    train_features = getAvgVecsFromIdx(train_features_idx)
    test_features = getAvgVecsFromIdx(test_features_idx)

    train_labels = np.array(train_labels).reshape(-1,1)
    test_labels = np.array(test_labels).reshape(-1,1)
    print "label is like:", train_labels[0]
    # run four classifiers
    runClassifiers(train_features, train_labels, test_features, test_labels) 








