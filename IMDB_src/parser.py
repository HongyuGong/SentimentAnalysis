import numpy as np
import pickle 
import os
from vocab import Vocab

EMBEDDING_PATH = '/projects/csl/viswanath/data/public/embedding/'
VOCAB_FILE = 'vocab.txt'
VECTOR_FILE = 'vectors.bin'

INPUT_PATH = '/projects/csl/viswanath/data/public/dumpedIMDBFolder/aclImdb'
OUTPUT_PATH = '/projects/csl/viswanath/data/public/dumpedIMDBFolder/IMDB'
CHECK_PATH = '/projects/csl/viswanath/data/public/dumpedStanfordFolder'

if __name__ == '__main__':

	os.system('mkdir -p %s' % OUTPUT_PATH)

	vocab = Vocab(directory = EMBEDDING_PATH,
				  vocabInputFile = VOCAB_FILE, 
				  vectorInputFile = VECTOR_FILE)

	
	dataset_labels = ['train', 'test']

	for dataset_label in dataset_labels:

		scores = list()
		idxs = list()
		
		for path in ['pos', 'neg']:
			
			path = '/'.join([INPUT_PATH, dataset_label, path])
			count = 0

			for fname in os.listdir(path):

				_, score = fname[:-4].split('_')
				fname = '/'.join([path, fname])
				raw = open(fname, 'r').read()

				idx = vocab.getIndex(raw)

				if len(idx) == 0:
					continue

				scores.append(score)
				idxs.append(idx)
				count += 1

				if count % 1000 == 0:
					print 'parsed %d samples in %s' % (count, path)

		n_feature = len(scores)
		scores = np.array(scores, dtype = np.float32)
		max_val = np.max(scores)
		min_val = np.min(scores)

		fname = '/'.join([OUTPUT_PATH, '_'.join([dataset_label, 'feature.pickle'])])
		pickle.dump(idxs, open(fname, 'wb'))

		fname = '/'.join([OUTPUT_PATH, '_'.join([dataset_label, 'label_binary_value.pickle'])])
		binary_threshold = (max_val + min_val) / 2.0
		binary_value = np.floor(scores / binary_threshold).astype(np.int8)
		pickle.dump(binary_value, open(fname, 'wb'))

		fname = '/'.join([OUTPUT_PATH, '_'.join([dataset_label, 'label_binary_vec.pickle'])])
		binary_vec = np.zeros((n_feature, 2))
		binary_vec[np.arange(n_feature), binary_value] = 1
		pickle.dump(binary_vec, open(fname, 'wb'))

		fname = '/'.join([OUTPUT_PATH, '_'.join([dataset_label, 'label_fiveClasses_vec.pickle'])])
		five_class_threshold = (max_val + min_val) / 5.0
		five_class_value = np.floor(scores / five_class_threshold).astype(np.int8)
		five_class_vec = np.zeros((n_feature, 5))
		five_class_vec[np.arange(n_feature), five_class_value] = 1
		pickle.dump(five_class_vec, open(fname, 'wb'))


	## sanity check

	for fname in os.listdir(OUTPUT_PATH):

		if 'feature' in fname:
			continue
 
		print '----- %s data shape -----' % fname
		print 'stanford:', pickle.load(open('/'.join([CHECK_PATH, fname]))).shape
		print 'imdb:', pickle.load(open('/'.join([OUTPUT_PATH, fname]))).shape






