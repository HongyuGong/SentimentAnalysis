"""
util functions:
dump data from dataseet
and load data
"""
import pickle
from vocab import Vocab
import numpy as np

# ------------------- public variable ----------------------
stanford_data_folder ="../data/stanfordSentimentTreebank/"
dumped_stanford_data_folder = "../data/dumpedStanfordFolder/"
vecDim = 300
directory = "../embedding/"
vocabInputFile = "vocab.txt"
vectorInputFile = "vectors.bin"
dumped_imdb_data_folder = "../data/dumpedIMDBFolder/IMDB/"
# ------------------- public variable ----------------------

def readStanfordLabel(sen_list):
    """
    input: a list of sentences, dictionary.txt (sen -> phrase_id), sentiment_labels.txt (phrase_id -> sentiment_values)
    """
    # read dictionary
    phrase_pid_dict = dict()
    f = open(stanford_data_folder+"dictionary.txt")
    for line in f.readlines():
        phrase, pid = line.strip().split("|")
	phrase = phrase.strip()
	phrase = " "+phrase+" "
        phrase = phrase.replace(" ( ", " -LRB- ")
        phrase = phrase.replace(" ) ", " -RRB- ")
	phrase = phrase.strip()
        pid =  int(pid)
        phrase_pid_dict[phrase] = pid
    f.close()
    
    # read sentiment_labels
    f = open(stanford_data_folder+"sentiment_labels.txt")
    pid_sentiment_dict = dict()
    lines = f.readlines()
    lines.pop(0)
    for line in lines:
        pid, sentiment = line.strip().split("|")
        pid = int(pid)
        sentiment_value = float(sentiment)
        pid_sentiment_dict[pid] = sentiment_value
    f.close()

    # match sentiment with the sentences
    sentiment_value_list = []
    sentiment_binary_value = []
    sentiment_binary_vec = []
    sentiment_fiveClasses_vec = []
    for sen in sen_list:
        if (sen not in phrase_pid_dict):
            print "non-exist:", sen
        else:
            pid = phrase_pid_dict[sen]
            sentiment_value = pid_sentiment_dict[pid]
            sentiment_value_list.append(sentiment_value)
	    # binary class
            sentiment_class = int(sentiment_value * 2)
	    if (sentiment_class == 2):
		sentiment_class = 1
	    sentiment_binary_value.append([sentiment_class])
	    binary_vec = [0] * 2
	    binary_vec[sentiment_class] = 1
	    sentiment_binary_vec.append(binary_vec[:])
	    # five classes
	    sentiment_class = int(sentiment_value * 5)
	    if (sentiment_class == 5):
		sentiment_class = 4
	    multi_vec = [0] * 5
	    multi_vec[sentiment_class] = 1
	    sentiment_fiveClasses_vec.append(multi_vec[:])
	    #try:
                #sentiment_oneHotVec[sentiment_class] = 1
                #sentiment_class_list.append([sentiment_class])
	    #except:
		#print "sentiment_value", sentiment_value, "sentiment_class", sentiment_class


    return sentiment_value_list, sentiment_binary_value, sentiment_binary_vec, sentiment_fiveClasses_vec


def readStanfordFeature(sen_list):
    """
    input: sen_list
    output: a list of lists of word idx in a sentence
    """
    vocab = Vocab(vecDim, directory, vocabInputFile, vectorInputFile)
    #feature = vocab.getAvgCxtVec(sen_list)
    feature = vocab.getContextIdList(sen_list) 
    return feature

def dumptStanfordDataset():
    """
    input: datasetSentences.txt
    output: dumped averaged representation, dumped labels
    """
    sen_list = []
    sen_file = stanford_data_folder+"datasetSentences.txt"
    f = open(sen_file, "r")
    lines = f.readlines()
    lines.pop(0)
    for line in lines:
        seq = line.strip()
        sen_id, sen = seq.split('\t')
        sen_list.append(sen)
    f.close()

    # read datasetSplit
    category_list = []
    f = open(stanford_data_folder+"datasetSplit.txt", "r")
    lines = f.readlines()
    lines.pop(0)
    for line in lines:
        sen_id, category = line.strip().split(",")
        sen_id = int(sen_id)
        category = int(category)
        category_list.append(category)
    f.close()
    #print "len category_list == len sentiment_value_list", len(category_list)==len(sentiment_value_list)

    #  feature as average vectors
    feature = readStanfordFeature(sen_list)
    #print "feature dimension should be 300", len(feature[0])
    
    # label
    #sentiment_value_list, sentiment_class_list = readStanfordLabel(sen_list)
    sentiment_value_list, sentiment_binary_value, sentiment_binary_vec, sentiment_fiveClasses_vec = readStanfordLabel(sen_list)
    print "len category_list == len sentiment_value_list", len(category_list)==len(sentiment_value_list)

    # split train features into train (1) /dev (3) /test (2)
    train_feature = np.array([feature[i] for i in range(len(feature)) if category_list[i] != 2])
    test_feature = np.array([feature[i] for i in range(len(feature)) if category_list[i] == 2])
    
    # split train labels into train/dev/test
    train_label_binary_value = np.array([sentiment_binary_value[i] for i in range(len(sentiment_binary_value)) if category_list[i] != 2])
    train_label_binary_vec = np.array([sentiment_binary_vec[i] for i in range(len(sentiment_binary_vec)) if category_list[i] != 2])
    train_label_fiveClasses_vec = np.array([sentiment_fiveClasses_vec[i] for i in range(len(sentiment_fiveClasses_vec)) if category_list[i] != 2])

    test_label_binary_value = np.array([sentiment_binary_value[i] for i in range(len(sentiment_binary_value)) if category_list[i] == 2])
    test_label_binary_vec = np.array([sentiment_binary_vec[i] for i in range(len(sentiment_binary_vec)) if category_list[i] == 2])
    test_label_fiveClasses_vec = np.array([sentiment_fiveClasses_vec[i] for i in range(len(sentiment_fiveClasses_vec)) if category_list[i] == 2])


    print "# of train_feature", len(train_feature), "# of train_label", len(train_label_binary_value)
    print "# of test_feature", len(test_feature), "# of test_label", len(test_label_binary_value)
    print "test_label dim:", len(test_label_binary_value[0])
    # dump features and labels
    with open(dumped_stanford_data_folder+"train_feature.pickle", "wb") as handle:
        pickle.dump(train_feature, handle)
    with open(dumped_stanford_data_folder+"train_label_binary_value.pickle", "wb") as handle:
        pickle.dump(train_label_binary_value, handle)
    with open(dumped_stanford_data_folder+"train_label_binary_vec.pickle", "wb") as handle:
        pickle.dump(train_label_binary_vec, handle)
    with open(dumped_stanford_data_folder+"train_label_fiveClasses_vec.pickle", "wb") as handle:
        pickle.dump(train_label_fiveClasses_vec, handle)


    with open(dumped_stanford_data_folder+"test_feature.pickle", "wb") as handle:
        pickle.dump(test_feature, handle)
    with open(dumped_stanford_data_folder+"test_label_binary_value.pickle", "wb") as handle:
        pickle.dump(test_label_binary_value, handle)
    with open(dumped_stanford_data_folder+"test_label_binary_vec.pickle", "wb") as handle:
        pickle.dump(test_label_binary_vec, handle)
    with open(dumped_stanford_data_folder+"test_label_fiveClasses_vec.pickle", "wb") as handle:
        pickle.dump(test_label_fiveClasses_vec, handle)


    print "finish dumping features and labels"
    

def loadStanfordDataset(num_classes, label_is_vec):
    
    # load train features and train labels
    with open(dumped_stanford_data_folder+"train_feature.pickle", "rb") as handle:
        train_feature = pickle.load(handle)

    # load test features and test labels
    with open(dumped_stanford_data_folder+"test_feature.pickle", "rb") as handle:
        test_feature = pickle.load(handle)

    if (num_classes == 2 and not label_is_vec):
	train_fn = "train_label_binary_value.pickle"
	test_fn = "test_label_binary_value.pickle"
    elif (num_classes == 2 and label_is_vec):
	train_fn = "train_label_binary_vec.pickle"
	test_fn = "test_label_binary_vec.pickle"
    else:
	train_fn = "train_label_fiveClasses_vec.pickle"
	test_fn = "test_label_fiveClasses_vec.pickle"

    with open(dumped_stanford_data_folder+train_fn, "rb") as handle:
	train_label = pickle.load(handle)
    with open(dumped_stanford_data_folder+test_fn, "rb") as handle:
        test_label = pickle.load(handle)

    return train_feature, train_label, test_feature, test_label
    
def loadIMDBDataset(num_classes, label_is_vec):

    # load train features and train labels
    with open(dumped_imdb_data_folder+"train_feature.pickle", "rb") as handle:
        train_feature = pickle.load(handle)

    # load test features and test labels
    with open(dumped_imdb_data_folder+"test_feature.pickle", "rb") as handle:
        test_feature = pickle.load(handle)

    if (num_classes == 2 and not label_is_vec):
        train_fn = "train_label_binary_value.pickle"
        test_fn = "test_label_binary_value.pickle"
    elif (num_classes == 2 and label_is_vec):
        train_fn = "train_label_binary_vec.pickle"
        test_fn = "test_label_binary_vec.pickle"
    else:
        train_fn = "train_label_fiveClasses_vec.pickle"
        test_fn = "test_label_fiveClasses_vec.pickle"

    with open(dumped_imdb_data_folder+train_fn, "rb") as handle:
        train_label = pickle.load(handle)
    with open(dumped_imdb_data_folder+test_fn, "rb") as handle:
        test_label = pickle.load(handle)

    return train_feature, train_label, test_feature, test_label

if __name__=="__main__":
    dumptStanfordDataset()








