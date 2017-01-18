# SentimentAnalysis

Sentiment Analysis

Dataset: Stanford Sentiment Dataset and IMDB.

Goal: apply different classifier for sentiment detection on two commonly-used datasets above. Classifiers used:
(1) Classic classifiers: inear classifier, logistic classifier, perceptron and SVM;
(2) Neural Network: 	single-layer feedforward neural network;
				RNN (with basic, GRU and LSTM units).

Code Structure:
Here is a list of source scripts:
	classifiers.py: implement simple classifiers like linear classifier, logistic classifier, perceptron and SVM.
	data_util.py: data preprocessing, feature dumping and loading.
	fnn_tf.py: implement single-layer feedforward neural network for sentiment classification.
	rnn_tf: implement recurrent neural network with basic, GRU and LSTM units.
	vocab.py: implement class Vocab for embedding processing.
	word2vec_tf.py: implement functions for word embedding training and saving.
	IMDB_src/parser.py: parse the raw data in IMDB review dataset.
	IMDB_src/vocab.py: transforms IMDB text into embedding.