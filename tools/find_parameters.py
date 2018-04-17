#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function          
import fileinput
import sys
import re
import os
import csv
import numpy as np
import collections
import keras
import util
import opt
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, GRU, Dense, Activation, Conv1D, MaxPooling1D, Flatten, TimeDistributed, Bidirectional
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras import backend as K

numColTag = 4
colIgnore = []
embeddingsArgument = dict()
nbFeat = 0
codeInterestingTags = []

longopts = ["ignoreColumns=", "columnOfTags=", "test=", "train=", "embeddings="]
shortopts = "i:t:e:"
filenameTrain = None
filenameTest = None
fileEmbed = None
numEmbed = None


usage_string = """\
Usage: {progname} OPTIONS <corpus>
Train an RNN

OPTIONS may be:

-i or --ignoreColumns <n-entities>
    To ignore some columns, and do not treat them as features

-t or --columnOfTags <AM.quantise>
    To give the number of the column containing tags (default, 4)
    Careful! The first column is number 0, the second number 1, ...

_e or --embeddings <n-entities>
    To give some files containing embeddings.
    First, you give the path of the file containing embeddings,
    and separate with a \",\" you gave the column concern by this file.
    eg: file1,2:file2,5
    Careful! You can't have a column in common with ignoreColumns. 

"""

def uniq(seq):
   # not order preserving
   set = {}
   map(set.__setitem__, seq, [])
   return set.keys()


def treat_options( opts, arg, n_arg, usage_string ) :
    """
        Callback function that handles the command line options of this script.
        
        @param opts The options parsed by getopts. Ignored.
        
        @param arg The argument list parsed by getopts.
        
        @param n_arg The number of arguments expected for this script.    
    """
    global numColTag
    global colIgnore
    global filenameTrain
    global filenameTest
    global embeddingsArgument
    global fileEmbed
    global numEmbed

    ctxinfo = util.CmdlineContextInfo(None, opts)
    util.treat_options_simplest(opts, arg, n_arg, usage_string)
    
    for o, a in ctxinfo.iter(opts):
        if o in ("--columnOfTags", "-t"):
            numColTag = ctxinfo.parse_uint(a)
        elif o in ("--ignoreColumns", "-i"):
            colIgnore = a.split(":")
            for i in range(len(colIgnore)):
                colIgnore[i] = int(colIgnore[i])
        elif o == "--train":
            filenameTrain = os.path.relpath(a)
        elif o == "--test":
            filenameTest = os.path.relpath(a)
        elif o in ("--embeddings", "-e"):
            embeddingsFileAndCol = a.split(":")
            for i in range(len(embeddingsFileAndCol)):
                embeddingsFileAndCol[i] = embeddingsFileAndCol[i].split(",")
                fileEmbed = embeddingsFileAndCol[i][0]
                numCol = embeddingsFileAndCol[i][1]
                if(embeddingsArgument.has_key(int(numCol))):
                   sys.stderr.write("Error with argument --embeddings")
                   exit()
                embeddingsArgument[int(numCol)] = fileEmbed
                numEmbed = int(numCol)
        else:
            raise Exception("Bad arg: " + o)
        
    if(filenameTrain == None or filenameTest == None):
        print("You need to give a train and test!")
        exit(1)
    colIgnore.append(numColTag)
    colIgnore = uniq(colIgnore)
    colIgnore.sort(reverse=True)
    
def enumdict():
    a = collections.defaultdict(lambda : len(a))
    return a
    
def init(line, features, vocab):
    global nbFeat
    curSequence = []
    for feati in range(nbFeat):  
        if(feati == numColTag):
            tagsCurSeq = []
        curSequence.append([])
        features.append([])
        vocab[feati]["0"] = 0
    return curSequence, features, vocab, tagsCurSeq
    
def handleEndOfSequence(tags, tagsCurSeq, features, curSequence):
    global nbFeat
    for feati in range(nbFeat):
        if(feati == numColTag):
            tags.append(tagsCurSeq)
            tagsCurSeq = []
        features[feati].append(curSequence[feati])
        curSequence[feati] = []
    return tags, tagsCurSeq, features, curSequence
    
def handleSequence(line, tagsCurSeq, vocab, curSequence):
    global nbFeat
    line = line.strip().split("\t")
    for feati in range(nbFeat):
        if(feati == numColTag):
            tagsCurSeq += [vocab[feati][line[feati]]]
        curSequence[feati] += [vocab[feati][line[feati]]]
    return tagsCurSeq, vocab, curSequence
    
def load_text(filename, vocab):
    global nbFeat
    start = True
    features = []
    tags = []
    with open(filename) as fp:
        for line in fp:
            if(nbFeat == 0):
                nbFeat = len(line.strip().split("\t"))
                vocab = collections.defaultdict(enumdict) 
            if(start == True):
                curSequence, features, vocab, tagsCurSeq = init(line, features, vocab)
                start = False
            if(line == "\n"):
                tags, tagsCurSeq, features, curSequence = handleEndOfSequence(tags, tagsCurSeq, features, curSequence) 
            else:
                tagsCurSeq, vocab, curSequence = handleSequence(line, tagsCurSeq, vocab, curSequence)
    return features, tags, vocab
    
def vectorize(features, tags, vocab, unroll):
    X_train = []
    for i in range(len(features)):
        feature = pad_sequences(features[i], unroll + 1, np.int32, 'post', 'post', 0)
        X_trainCurFeat = np.zeros((len(feature), unroll), dtype=np.int32)
        features[i] = feature
        X_train.append(X_trainCurFeat)
        
    tags = pad_sequences(tags, unroll + 1, np.int32, 'post', 'post', 0)
    Y_train = np.zeros((len(tags), unroll, 1), dtype=np.int32)
    sample_weight = np.zeros((len(tags), unroll), dtype=np.float64)
    
    mask = np.zeros(Y_train.shape)
    for feati in range(len(features)):
        for i in range(len(features[feati])):
            for j in range(unroll):
                X_train[feati][i,j] = features[feati][i,j]
    
    for i in range(len(tags)):
        for j in range(unroll):
            curTag = tags[i,j]
            Y_train[i,j,0] = curTag
            if(curTag in codeInterestingTags):
                sample_weight[i][j] = 1.0
            else:
                sample_weight[i][j] = 0.01
            if(Y_train[i,j,0] != 0):
                mask[i,j,0] = 1
    for i in colIgnore:
        X_train.pop(i)
    
    return X_train, Y_train, mask, sample_weight
    

def make_modelMWE(Config, num_tags, unroll, vocab, embedding_matrix, dimension):
    inputs = []
    embeddings = []
    for i in range(nbFeat):
        if(i in colIgnore):
            continue
        nameInputFeat = 'Column'+str(i)
        inputFeat = Input(shape=(unroll,), dtype='int32', name=nameInputFeat)
        inputs.append(inputFeat)
        if(embeddingsArgument.has_key(i)):
            x = Embedding(output_dim=dimension, input_dim=len(vocab[i]), weights=[embedding_matrix], input_length=unroll, trainable=True)(inputFeat)
        else:    
            x = Embedding(output_dim=Config.embed, input_dim=len(vocab[i]), input_length=unroll, trainable=True)(inputFeat)
        embeddings.append(x)
    x = keras.layers.concatenate(embeddings)
    x = Bidirectional(GRU(Config.hidden, return_sequences=True))(x)
    x = TimeDistributed(Dense(num_tags))(x)
    x = Activation('softmax')(x)
    model = Model(inputs = inputs, outputs = [x])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['acc'], sample_weight_mode="temporal") ###############################
    return model
        
def maxClasses(classes, Y_test, unroll, mask):
    prediction = np.zeros(Y_test.shape)
    for i in range(len(Y_test)):
        for j in range(unroll-1):
            if(mask[i][j] == 0):
                prediction[i][j][0] = 0
            else:
                maxTag = np.argmax(classes[i][j])
                if(maxTag == 0):
                    classes[i][j][0] = 0
                    maxTag = np.argmax(classes[i][j])
                prediction[i][j][0] = maxTag
        
    return prediction

def genereTag(prediction, vocab, unroll):
    rev_vocabTags = {i: char for char, i in vocab[numColTag].items()}
    for i in range(len(prediction)):
        for j in range(unroll-1):
            curTagEncode = prediction[i][j][0]
            if(curTagEncode == 0):
                break
            else:
                print(rev_vocabTags[curTagEncode])
        print()
        
def loadEmbeddings(vocab, filename, numColEmbed):
    readFirstLine = True
    print('loading embeddings from "%s"' % filename, file=sys.stderr)
    with open(filename) as fp:
        for line in fp:
            tokens = line.strip().split(' ')
            if(readFirstLine):
                lenVocab = int(tokens[0])+len(vocab[numColEmbed])+1
                dimension = int(tokens[1])
                embedding = np.zeros((lenVocab, dimension), dtype=np.float32)
                readFirstLine = False
            else:
                word = tokens[0]
                if word in vocab[numColEmbed]:
                    lenVocab-=1       
                embedding[vocab[numColEmbed][word]] = [float(x) for x in tokens[1:]]
                #print("never seen! lenVocab : ",lenVocab," ",len(vocab[numColEmbed]))
    #np.reshape(embedding, (lenVocab, dimension))

    embedding = np.delete(embedding, list(range(lenVocab-1, len(embedding))), 0)
    
    return embedding, vocab, dimension    
    
# define a parameter config
class Config(opt.Config):
    batch = opt.Choice([16, 32, 64], default=32)
    epochs = 1
    hidden = opt.Choice([128, 250], default=128)
    embed = opt.Choice([6, 8, 10], default=8)
    
def main():
    global codeInterestingTags
    args = util.read_options(shortopts, longopts, treat_options, -1, usage_string )
    
    unroll = 128
    vocab = []
    
    sys.stderr.write("Load training file..\n")
    features, tags, vocab = load_text(filenameTrain, vocab)    
    
    X_train, Y_train, mask, sample_weight = vectorize(features, tags, vocab, unroll)
    sys.stderr.write("Load testing file..\n")
    features, tags, vocab = load_text(filenameTest, vocab)
    X_valid, Y_valid, mask, useless = vectorize(features, tags, vocab, unroll)
    
    num_tags = len(vocab[numColTag])
    iterations = 1000
        
    embedding_matrix, vocab, dimension = loadEmbeddings(vocab, fileEmbed, numEmbed)
    
    sys.stderr.write('training\n')
    for i in range(iterations):
        opt.generate(Config)
        model = make_modelMWE(Config, num_tags, unroll, vocab, embedding_matrix, dimension)
        for epoch in range(10):
            model.fit(X_train, Y_train, epochs=1, verbose=0, batch_size=Config.batch, shuffle=True)
            Config.epochs = epoch + 1
            loss = model.evaluate(X_valid, Y_valid, batch_size=Config.batch, verbose=0)
            print('PERF', loss[-1], Config)
            sys.stdout.flush()
            
    
    
main()
