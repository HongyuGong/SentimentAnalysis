import os
import tensorflow as tf
from tensorflow.models.embedding import word2vec
import argparse
import numpy as np
from array import array

flags = tf.app.flags
FLAGS = flags.FLAGS

def trainWord2Vec(train_data, eval_data, save_path, vec_dim):
    FLAGS.train_data = train_data
    FLAGS.eval_data = eval_data
    FLAGS.save_path = save_path
    FLAGS.embedding_size = vec_dim
    
    word2vec.main([])
    print "done training data"

def saveVec(model_path, emb_path):
    embeddings = tf.contrib.framework.load_variable(model_path, 'emb')
    embeddings = embeddings.reshape(-1)
    output_file = open(emb_path+"vectors.bin", "wb")
    a = array('f', embeddings)
    a.tofile(output_file)
    output_file.close()
    print "finish writing vectors.bin..."
        

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default="../data/text8", type=str)
    parser.add_argument('--eval', default="../data/eval", type=str)
    parser.add_argument('--modelPath', default="../data/model/", type=str)
    parser.add_argument('--embPath', default="../data/model/", type=str)
    parser.add_argument('--dim', default=300, type=int)
    args = parser.parse_args()
    train_data = args.train
    eval_data = args.eval
    model_path = args.modelPath
    emb_path = args.embPath
    vec_dim = args.dim

    # model training
    trainWord2Vec(train_data, eval_data, model_path, vec_dim)

    # save embedding
    saveVec(model_path, emb_path)


    


    

















    
