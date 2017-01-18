import numpy as np
import sys
import array
import os
import struct
import re
import nltk

def normalizeMatrix(mat):
	matNorm = np.zeros(mat.shape)
	d = (np.sum(mat ** 2, 1) ** (0.5))
	matNorm = (mat.T / d).T

	return matNorm

def dist(s1, s2, label):

	cross = np.dot(s1, s2.T)
	_, s, _ = np.linalg.svd(cross)
	theta = np.arccos(s)
	
	if label.startswith('all'):

		start = 0
		order = int(label.split('_')[-1])

		return np.average(s[start:] ** order) ** (1.0 / order)

	elif 'toend' in label:

		start = int(label[0])
		order = int(label.split('_')[-1])

		if len(s) > start:
			return np.average(s[start:] ** order) ** (1.0 / order)
		else:
			return np.average(s[:] ** order) ** (1.0 / order)

	elif label == 'max':

		return s[0]

	elif label == 'min':

		return s[-1]

	# elif label == 'grassmann':

	# 	return np.average(theta ** 2) ** 0.5

	# elif label == 'asimov':

	# 	return theta[-1]

	# elif label == 'binent-cauchy':

	# 	return (1 - np.prod(np.cos(theta) ** 2)) ** 0.5

	# elif label == 'chordal':

	# 	return np.average(np.sin(theta) ** 2) ** 0.5

	# elif label == 'fubini-study':

	# 	return np.arccos(np.prod(np.cos(theta)))

	# elif label == 'martin':

	# 	return np.log(np.prod((1/np.cos(theta)) ** 2)) ** 0.5

	# elif label == 'procrustes':

	# 	return np.average(np.sin(theta/2) ** 2) ** 0.5

	# elif label == 'projection':

	# 	return np.sin(theta[-1])

	# elif label == 'spectral':

	# 	return np.sin(theta[-1]/2)


def cosSim(array1, array2):

	return np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2))

class Vocab:

	def __init__(self, directory, vocabInputFile, vectorInputFile, vecDim = 300, corpusDir = '', corpusName = '',
		                funcList = []):

		self.vecDim = vecDim
		self.directory = directory
		self.corpusDir = directory + corpusDir
		self.corpusName = directory + corpusName
		self.vocabFile = directory + vocabInputFile
		self.vecFile = directory + vectorInputFile

		self.readVocabFromFile()

		self.readFuncWords(funcList)

	def setParams(self, pcaRank, window = -1, norm = 0, label = 'all_2', weight = float('inf'), remove = 0, zeromean = 0):

		self.pcaRank = pcaRank
		self.window = window
		self.label = label

		if norm != 0:

			self.vecMatrix = normalizeMatrix(self.vecMatrix)

		if zeromean != 0:

			mean = np.mean(self.vecMatrix, axis = 0, keepdims = True)
			self.vecMatrix -= mean

		if weight < float('inf') and weight > 0:

			weight = weight / (weight + self.vocabFreq)
			for i in xrange(len(weight)):
				self.vecMatrix[i] *= weight[i]

		if remove >= 1:

			pca = PCA(n_components = remove)
			pca.fit(self.vecMatrix)

			for i in xrange(int(remove)):
				
				vecCommon = pca.components_[i]
				self.vecMatrix -= np.outer(np.dot(self.vecMatrix, vecCommon), vecCommon)

		elif remove > 0:

			pca = PCA()
			pca.fit(self.vecMatrix)

			k = 0
			while True:
				if pca.explained_variance_ratio_[k] > remove:
					break
				k += 1

			for i in xrange(k):
				vecCommon = pca.components_[i]
				self.vecMatrix -= np.outer(np.dot(self.vecMatrix, vecCommon), vecCommon)
			

	def readFuncWords(self, funcList):

		funcWords = set()

		if 'freq' in funcList:
			num = int(funcList[-1])
			for i in xrange(num):
				funcWords.add(self.vocabList[i])

		elif 'all' in funcList:
			for path in FUNCWORD_LIST.values():
				f = open(FUNCWORD_PATH + path, 'r')
			for line in f.readlines():
				funcWords.add(line.rstrip())

		else:
			for label in funcList:
				try:
					f = open(FUNCWORD_PATH + FUNCWORD_LIST[label], 'r')
				except:
					print label
					continue
				for line in f.readlines():
					funcWords.add(line.rstrip())

		self.funcWords = funcWords

	def readVocabFromFile(self):

		if not (os.path.isfile(self.vocabFile) and os.path.isfile(self.vecFile)):
			print 'embedding does not exist.'
			os.system('rm -r %s' % self.corpusName)
			os.system('for file in %s*/*; do cat $file >> %s; done' % (self.corpusDir, self.corpusName))
			os.system('%s %s %s %s %d 100' % (WORD2VEC_SCRIPT, self.corpusName, self.vecFile, self.vocabFile, self.vecDim))


		vocabList = list()
		vocabIndex = dict()
		vocabCount = list()

		f = open(self.vocabFile, "r")
		idx = 0
		for line in f.readlines():
			raw = line.lower().split()
			vocabList.append(raw[0])
			vocabCount.append(int(raw[1]))
			vocabIndex[raw[0]] = idx 
			idx += 1

		self.vocabList = vocabList
		self.vocabIndex = vocabIndex
		self.vocabSize = len(self.vocabList)
		self.vocabCount = np.array(vocabCount)
		self.vocabFreq = self.vocabCount / float(np.sum(vocabCount))


		vecDim = self.vecDim
		vecMatrix = array.array('f')
		vecMatrix.fromfile(open(self.vecFile, 'rb'), self.vocabSize * vecDim)
		vecMatrix = np.reshape(vecMatrix, (self.vocabSize, vecDim))[:, 0:vecDim]
		self.vecMatrix = vecMatrix


	def readPSLVocabFromFile(self):

		vocabFile = self.directory + 'vocab.txt'
		countDict = dict()

		f = open(self.vocabFile, "r")
		for line in f.readlines():
			raw = line.lower().split()
			countDict[raw[0]] = int(raw[1])

		vocabFile = self.directory + 'vectors.txt'
		vocabList = list()
		vocabIndex = dict()
		vecMatrix = list()
		vocabCount = list()

		ind = 0
		for raw in open(vocabFile, 'r').readlines():
			raw = raw.rstrip().split()
			word = raw[0]
			vec = np.array(map(float, raw[1:]))
			try:
				vocabCount.append(countDict[raw[0]])
			except:
				continue
			vocabList.append(ind)
			vocabIndex[word] = ind
			vecMatrix.append(vec)
			ind += 1

		vecMatrix = np.array(vecMatrix)

		self.vocabList = vocabList
		self.vocabIndex = vocabIndex
		self.vocabSize = len(self.vocabList)
		self.vocabCount = np.array(vocabCount)
		self.vocabFreq = self.vocabCount / float(np.sum(vocabCount))
		self.vecMatrix = vecMatrix

	def computeSenSim(self, sen1, sen2):

		indList1 = self.getIndex(sen1)
		indList2 = self.getIndex(sen2)

		if len(indList1) * len(indList2) == 0:
			return -float('inf'), 0, 0

		if np.abs(self.pcaRank) == 0:
			avg1 = self.getAverage(indList1)
			avg2 = self.getAverage(indList2)
			return cosSim(avg1, avg2), len(indList1), len(indList2)

		else:
			s1, ratio1 = self.getSubspace(indList1)
			s2, ratio2 = self.getSubspace(indList2)
			return dist(s1, s2, self.label), len(indList1), len(indList2)


	def getIndex(self, senStr):

		wordList = nltk.word_tokenize(senStr.rstrip().lower().decode('utf-8'))
		indList = list()
		for word in wordList:
			if word in self.funcWords:
				continue
			try:
				indList.append(self.vocabIndex[word])
			except:
				continue

		return indList

	def getAverage(self, indList):

		return np.average(self.vecMatrix[indList], axis = 0)

	def getSubspace(self, indList):

		# print 'after:', np.linalg.norm(np.average(self.vecMatrix, axis = 0))

		if self.pcaRank >= 1:
			self.pcaRank = int(self.pcaRank)
			pca = PCA(n_components = self.pcaRank)
			pca.fit(self.vecMatrix[np.array(indList)])
			# ratio = np.zeros(10)
			# ratio[:len(pca.explained_variance_ratio_)] = pca.explained_variance_ratio_

			return pca.components_[:self.pcaRank], pca.explained_variance_ratio_[:self.pcaRank]

		elif self.pcaRank > 0:

			pca = PCA()
			pca.fit(self.vecMatrix[np.array(indList)])
			ratio = pca.explained_variance_ratio_
			k = 2
			while True:
				if np.sum(ratio[:k]) > self.pcaRank:
					break
				k += 1

			return pca.components_[:k], pca.explained_variance_ratio_[:k]

		else:

			pca = PCA()
			pca.fit(self.vecMatrix[np.array(indList)])

			return pca.components_, pca.explained_variance_ratio_


	
