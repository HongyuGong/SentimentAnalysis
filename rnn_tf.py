"""
tensorflow implemention of RNN and LSTM
"""
import tensorflow as tf
import numpy as np
import data_util
from vocab import Vocab


def vanillaRNNCell(x, timesteps, weights, bias, input_dim, hidden_dim):

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, input_dim])
    x = tf.split(0, timesteps, x)
    
    rnn_unit = tf.nn.rnn_cell.BasicRNNCell(hidden_dim)
    outputs, states = tf.nn.rnn(rnn_unit, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights)+bias

def gruCell(x, timesteps, weights, bias, input_dim, hidden_dim):
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, input_dim])
    x = tf.split(0, timesteps, x)
    
    rnn_unit = tf.nn.rnn_cell.GRUCell(hidden_dim)
    outputs, states = tf.nn.rnn(rnn_unit, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights)+bias

def lstmCell(x, timesteps, weights, bias, input_dim, hidden_dim):

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, input_dim])
    x = tf.split(0, timesteps, x)

    rnn_unit = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, forget_bias=1.0)
    outputs, states = tf.nn.rnn(rnn_unit, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights)+bias
    
def classification(train_x, train_y, test_x, test_y, input_dim, max_timesteps, hidden_dim, output_dim, network_type, \
                   learning_rate=1e-3, max_iterations=100, batch_size=64):
    # variables for training
    x = tf.placeholder("float", [None, max_timesteps, input_dim])
    y = tf.placeholder("float", [None, output_dim])
    weights = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
    bias = tf.Variable(tf.random_normal([output_dim]))
    if (network_type=="vanilla"):
        pred_y = vanillaRNNCell(x, max_timesteps, weights, bias, input_dim, hidden_dim)
    elif (network_type == "gru"):
        pred_y = gruCell(x, max_timesteps, weights, bias, input_dim, hidden_dim)
    else: # network_type=="lstm"
        pred_y = lstmCell(x, max_timesteps, weights, bias, input_dim, hidden_dim)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_y, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # variables for testing
    correct_pred = tf.equal(tf.argmax(pred_y, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # initialization
    init = tf.initialize_all_variables()
    # batched training
    with tf.Session() as sess:
        sess.run(init)
        step = 0
        train_acc_list = []
        step_list = []
        for epoch in range(max_iterations):
            num_iter = int(np.ceil(len(train_x) / batch_size))
            for ind in range(num_iter):
                start = ind * batch_size
                end = min((ind+1)*batch_size, len(train_x))
                batch_x = train_x[start: end]
                batch_x = batch_x.reshape((len(batch_x), max_timesteps, input_dim))
                batch_y = train_y[start: end]
                #print "batch shape", batch_x.shape, batch_y.shape
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                
        # train accuracy
        train_x = train_x.reshape((-1, max_timesteps, input_dim))
        train_acc = sess.run(accuracy, feed_dict={x:train_x, y: train_y})
        print "train accuracy: %f" % train_acc

        # test accuracy
        test_x = test_x.reshape((-1, max_timesteps, input_dim))
        test_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
        print "test accuracy: %f" % test_acc


def getVecsFromIdx(train_idx_list, test_idx_list):
    vecDim = 300
    directory = "../embedding/"
    vocabInputFile = "vocab.txt"
    vectorInputFile = "vectors.bin"
    vocab = Vocab(vecDim, directory, vocabInputFile, vectorInputFile)

    train_vec_list = vocab.getVecFromId(train_idx_list)
    test_vec_list = vocab.getVecFromId(test_idx_list)
    #print "vec_list dim", len(vec_list[0][0]),"should be", 300
    return train_vec_list, test_vec_list

def padFeatures(features, input_dim, max_timesteps):
    # flatten and pad features: feature dim -- vecDim * max_timesteps
    for i in range(len(features)):
        vec = features[i]
        vec = list(vec.reshape(-1))
        if (len(vec) > input_dim * max_timesteps):
            vec = vec[0: input_dim * max_timesteps]
        else:
            vec.extend([0]*(input_dim*max_timesteps - len(vec)))
        features[i] = vec
    return np.array(features)

if __name__=="__main__":

    input_dim = 300
    output_dim = 5
    hidden_dim = 50
    max_timesteps = 50

##    """ stanford snetiment treebank """
##    train_features_idx, train_multi_labels, test_features_idx, test_multi_labels = data_util.loadStanfordDataset(5, True)
##    train_features, test_features = getVecsFromIdx(train_features_idx, test_features_idx)

    """ IMDB review """
    train_features_idx, train_multi_labels, test_features_idx, test_multi_labels = data_util.loadIMDBDataset(5, True)
    # downsample train dataset
    inds = np.random.choice(len(train_features_idx), size=min(len(train_features_idx), 8000))
    train_features_idx = [train_features_idx[i] for i in inds]
    train_multi_labels = np.array([train_multi_labels[i] for i in inds])
    # downsample test dataset
    test_inds = np.random.choice(len(test_features_idx), size=min(len(test_features_idx), 2000))
    test_features_idx = [test_features_idx[i] for i in test_inds]
    test_multi_labels = np.array([test_multi_labels[i] for i in test_inds])
    
    train_features, test_features = getVecsFromIdx(train_features_idx, test_features_idx)

    print "finish loading features ..."
    
    """ pad features to keep dimension consistency """
    train_features = padFeatures(train_features, input_dim, max_timesteps)
    test_features = padFeatures(test_features, input_dim, max_timesteps)
    print "train_features shape", train_features.shape, "test_features shape", test_features.shape
    print "train_label shape", train_multi_labels.shape, "test_label shape", test_multi_labels.shape

    """
    Specify two parameters: network_type
    network_type can be: "vanilla", "lstm", "gru"
    """
    network_type = "lstm" # "vanilla" or "gru" or "lstm"
    classification(train_features, train_multi_labels, test_features, test_multi_labels, input_dim, max_timesteps, hidden_dim, output_dim, network_type)











    
