#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import collections
import os
import sys

import keras
import numpy as np
from keras.layers import Embedding, Input, GRU, Dense, Activation, TimeDistributed, \
    Bidirectional
from keras.models import Model, model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model

import util
from reader import ReaderCupt

parser = argparse.ArgumentParser(description="""
        
        """)
parser.add_argument("--ignoreColumns", dest="ignoreColumns",
                    required=True, nargs='+', type=int,
                    help="""
                    To ignore some columns, and do not treat them as features.
                    """)
parser.add_argument("--columnOfTags", type=int,
                    required=True, dest='columnOfTags', default=4,
                    help="""
                    To give the number of the column containing tags (default, 4)
                    Careful! The first column is number 0, the second number 1, ...
                    """)
parser.add_argument("--embeddings", nargs='+', type=str, dest="embeddingsArgument",
                    help="""
                    To give some files containing embeddings.
                    First, you give the path of the file containing embeddings,
                    and separate with a \",\" you gave the column concern by this file.
                    eg: file1,2 file2,5
                    Careful! You can't have a column in common with ignoreColumns.
                    """)

parser.add_argument("--file", metavar="filename", dest="filename", required=True, type=argparse.FileType('r'),
                    help="""
                    Give a file in the Extended CoNLL-U (.cupt) format.
                    You can only give one file to train/test a model.
                    """)
parser.add_argument("--mode", type=str, dest='mode', required=True,
                    help="""
                    If the file is a train file and you want to create a model.
                    """)
parser.add_argument("--model", action='store', type=str,
                    required=True, dest='model',
                    help="""
                    Name of the model which you want to save/load without extension.
                    """)
parser.add_argument("--bio", action='store_const', const=True,
                    dest='bio',
                    help="""
                    Option to use the representation of BIO.
                    You can combine with other options like --gap or/and --vmwe.
                    You can't combine with --io option.
                    """)
parser.add_argument("--io", action='store_const', const=True,
                    dest='io',
                    help="""
                    Option to use the representation of BIO.
                    You can combine with other options like --gap or/and --vmwe.
                    You can't combine with --bio option.
                    """)
parser.add_argument("-g", "--gap", action='store_const', const=True,
                    dest='gap',
                    help="""
                    Option to use the representation of BIO/IO with gap.
                    """)
parser.add_argument("-mwe", "--category", action='store_const', const=True,
                    dest='withMWE',
                    help="""
                    Option to use the representation of BIO/IO with VMWE.
                    """)
parser.add_argument("--batch_size", required=True ,type=int,
                    dest='batch_size',
                    help="""
                    Option to initialize the size of batch for the RNN.
                    """)
parser.add_argument("--overlaps", action='store_const', const=True, dest='withOverlaps',
                    help="""
                    Option to use the representation of BIO/IO with overlaps.
                    We can't load a file test with overlaps.
                    By default, if option test and overlaps are activated, only the option test is considered. 
                    """)

numColTag = 4
colIgnore = []
embeddingsArgument = dict()
nbFeat = 0
codeInterestingTags = []
filename = None
isTrain = False
isTest = False
filenameModelWithoutExtension = None
FORMAT = None


def uniq(seq):
    # not order preserving
    set = {}
    map(set.__setitem__, seq, [])
    return set.keys()


def treat_options(args):
    global numColTag
    global colIgnore
    global filename
    global filenameModelWithoutExtension
    global embeddingsArgument
    global isTrain
    global isTest
    global FORMAT

    numColTag = args.columnOfTags
    colIgnore = args.ignoreColumns
    filename = args.filename
    filenameModelWithoutExtension = args.model

    if args.embeddingsArgument:
        embeddingsFileAndCol = args.embeddingsArgument
        for i in range(len(embeddingsFileAndCol)):
            embeddingsFileAndCol[i] = embeddingsFileAndCol[i].split(",")
            fileEmbed = embeddingsFileAndCol[i][0]
            numCol = embeddingsFileAndCol[i][1]
            if embeddingsArgument.has_key(int(numCol)):
                sys.stderr.write("Error with argument --embeddings")
                exit()
            embeddingsArgument[int(numCol)] = fileEmbed

    if args.mode.lower() == "train":
        isTrain = True
        isTest = False
    elif args.mode.lower() == "test":
        isTrain = False
        isTest = True
    else:
        sys.stderr.write("Error with argument --mode (train/test")
        exit(-10)

    if args.bio:
        FORMAT = "BIO"
    if args.io:
        FORMAT = "IO"

    if FORMAT is None:
        FORMAT = "BIO"

    if args.gap:
        FORMAT += "g"

    if args.withMWE:
        FORMAT += "cat"

    colIgnore.append(numColTag)
    colIgnore = uniq(colIgnore)
    colIgnore.sort(reverse=True)


def enumdict():
    a = collections.defaultdict(lambda: len(a))
    a["<unk>"] = 1
    return a


def init(line, features, vocab):
    global nbFeat
    curSequence = []
    for feati in range(nbFeat):
        if (feati == numColTag):
            tagsCurSeq = []
        curSequence.append([])
        features.append([])
        vocab[feati]["0"] = 0
    return curSequence, features, vocab, tagsCurSeq


def handleEndOfSequence(tags, tagsCurSeq, features, curSequence):
    global nbFeat
    for feati in range(nbFeat):
        if (feati == numColTag):
            tags.append(tagsCurSeq)
            tagsCurSeq = []
        features[feati].append(curSequence[feati])
        curSequence[feati] = []
    return tags, tagsCurSeq, features, curSequence


def handleSequence(line, tagsCurSeq, vocab, curSequence):
    global nbFeat
    line = line.strip().split("\t")
    for feati in range(nbFeat):
        if (feati == numColTag):
            tagsCurSeq += [vocab[feati][line[feati]]]
        curSequence[feati] += [vocab[feati][line[feati]]]

    return tagsCurSeq, vocab, curSequence


def load_text(filename, vocab):
    global nbFeat
    start = True
    features = []
    tags = []
    for sentence in filename:
        for line in sentence:
            if (nbFeat == 0):
                nbFeat = len(line.strip().split("\t"))
                vocab = collections.defaultdict(enumdict)
            if (start == True):
                curSequence, features, vocab, tagsCurSeq = init(line, features, vocab)
                start = False
            if (line == "\n"):
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
                X_train[feati][i, j] = features[feati][i, j]

    for i in range(len(tags)):
        for j in range(unroll):
            curTag = tags[i, j]
            Y_train[i, j, 0] = curTag
            if (curTag in codeInterestingTags):
                sample_weight[i][j] = 1.0
            else:
                sample_weight[i][j] = 0.01
            if (Y_train[i, j, 0] != 0):
                mask[i, j, 0] = 1

    for i in colIgnore:
        X_train.pop(i)

    return X_train, Y_train, mask, sample_weight


def make_modelMWE(hidden, embed, num_tags, unroll, vocab):
    inputs = []
    embeddings = []
    for i in range(nbFeat):
        if (i in colIgnore):
            continue
        nameInputFeat = 'Column' + str(i)
        inputFeat = Input(shape=(unroll,), dtype='int32', name=nameInputFeat)
        inputs.append(inputFeat)
        if (embeddingsArgument.has_key(i)):
            embedding_matrix, vocab, dimension = loadEmbeddings(vocab, embeddingsArgument[i], i)
            x = Embedding(output_dim=dimension, input_dim=len(vocab[i]), weights=[embedding_matrix],
                          input_length=unroll, trainable=True)(inputFeat)
        else:
            x = Embedding(output_dim=embed, input_dim=len(vocab[i]), input_length=unroll, trainable=True)(inputFeat)
        embeddings.append(x)
    x = keras.layers.concatenate(embeddings)
    x = Bidirectional(GRU(hidden, return_sequences=True))(x)
    x = TimeDistributed(Dense(num_tags))(x)
    x = Bidirectional(GRU(hidden, return_sequences=True))(x)
    x = TimeDistributed(Dense(num_tags))(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=[x])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['acc'],
                  sample_weight_mode="temporal")  ###############################
    return model


def maxClasses(classes, Y_test, unroll, mask):
    prediction = np.zeros(Y_test.shape)

    for i in range(len(Y_test)):
        for j in range(unroll - 1):
            if (mask[i][j] == 0):
                prediction[i][j][0] = 0
            else:
                maxTag = np.argmax(classes[i][j])
                if (maxTag == 0):
                    classes[i][j][0] = 0
                    maxTag = np.argmax(classes[i][j])
                    print(maxTag, i, j)
                prediction[i][j][0] = maxTag

        print()
    return prediction


def genereTag(prediction, vocab, unroll):
    rev_vocabTags = {i: char for char, i in vocab[numColTag].items()}
    pred = []
    listNbToken = []
    print(rev_vocabTags)
    for i in range(len(prediction)):
        nbToken = 0
        tag = ""

        for j in range(unroll - 1):
            curTagEncode = prediction[i][j][0]
            if curTagEncode != 0:
                nbToken += 1
                tag += rev_vocabTags[curTagEncode]
                pred.append(rev_vocabTags[curTagEncode])
        listNbToken.append(nbToken)
    return pred, listNbToken


def loadEmbeddings(vocab, filename, numColEmbed):
    readFirstLine = True
    print('loading embeddings from "%s"' % filename, file=sys.stderr)
    with open(filename) as fp:
        for line in fp:
            tokens = line.strip().split(' ')
            if (readFirstLine):
                lenVocab = int(tokens[0]) + len(vocab[numColEmbed]) + 1
                dimension = int(tokens[1])
                embedding = np.zeros((lenVocab, dimension), dtype=np.float32)
                readFirstLine = False
            else:
                word = tokens[0]
                if word in vocab[numColEmbed]:
                    lenVocab -= 1
                embedding[vocab[numColEmbed][word]] = [float(x) for x in tokens[1:]]
                # print("never seen! lenVocab : ",lenVocab," ",len(vocab[numColEmbed]))
    # np.reshape(embedding, (lenVocab, dimension))

    embedding = np.delete(embedding, list(range(lenVocab - 1, len(embedding))), 0)

    return embedding, vocab, dimension


def main():
    global codeInterestingTags
    args = parser.parse_args()

    treat_options(args)

    hidden = 512
    batch = args.batch_size
    unroll = batch
    embed = 64
    epochs = 10
    vocab = []

    sys.stderr.write("Load FORMAT ..\n")
    reformatFile = ReaderCupt(FORMAT, args.withOverlaps, isTest, filename)
    reformatFile.read()

    if isTrain:

        sys.stderr.write("Load training file..\n")
        features, tags, vocab = load_text(reformatFile.resultSequences, vocab)

        X, Y, mask, sample_weight = vectorize(features, tags, vocab, unroll)
        num_tags = len(vocab[numColTag])

        sys.stderr.write("Create model..\n")
        model = make_modelMWE(hidden, embed, num_tags, unroll, vocab)
        # plot_model(model, to_file='modelMWE.png', show_shapes=True)

        sys.stderr.write("Starting training...")
        model.fit(X, Y, batch_size=batch, epochs=epochs, shuffle=True,
                  sample_weight=sample_weight)

        sys.stderr.write("Save vocabulary...\n")
        # TODO - save vocab
        reformatFile.saveVocab(filenameModelWithoutExtension + ".voc", vocab)

        sys.stderr.write("Save model..\n")
        # serialize model to json
        model_json = model.to_json()
        with open(filenameModelWithoutExtension + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(filenameModelWithoutExtension + ".h5")

        sys.stderr.write("END training\n")

    elif isTest:


        sys.stderr.write("Load vocabulary...\n")
        # TODO - load vocab
        vocab = reformatFile.loadVocab(filenameModelWithoutExtension + ".voc")
        reformatFile.verifyUnknowWord(vocab)

        sys.stderr.write("Load model..\n")

        # Use statefull GRU with SGRU
        # load json and create model
        json_file = open(filenameModelWithoutExtension + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(filenameModelWithoutExtension + ".h5")

        model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['acc'],
                      sample_weight_mode="temporal")

        sys.stderr.write("Load testing file..\n")
        features, tags, useless = load_text(reformatFile.resultSequences, vocab)
        X, Y, mask, sample_weight = vectorize(features, tags, vocab, unroll)

        # model.evaluate(X, Y)
        classes = model.predict(X)
        # sys.stderr.write(classes.shape+ "\nclasses: "+ classes)
        prediction = maxClasses(classes, Y, unroll, mask)
        # nbErrors = np.sum(prediction != Y)
        # nbPrediction = np.sum(mask == 1)
        # acc = (nbPrediction - nbErrors) * 100 / float(nbPrediction)
        # sys.stderr.write(nbErrors nbPrediction)
        # sys.stderr.write("%.2f" % acc)
        # sys.stderr(str(prediction))

        pred, listNbToken = genereTag(prediction, vocab, unroll)
        print(len(pred))
        reformatFile.addPrediction(pred, listNbToken)

        # print(len(pred))
        sys.stderr.write("END testing\n")

    else:
        sys.stderr("Error argument: Do you want to test or train ?")
        exit(-2)


main()
