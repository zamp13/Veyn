#! /usr/bin/env python
# -*- coding:UTF-8 -*-

################################################################################
#
# Copyright 2010-2014 Carlos Ramisch, Vitor De Araujo, Silvio Ricardo Cordeiro,
# Sandra Castellanos
#
# candidates.py is part of mwetoolkit
#
# mwetoolkit is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# mwetoolkit is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with mwetoolkit.  If not, see <http://www.gnu.org/licenses/>.
#
################################################################################
from __future__ import print_function

import argparse
import collections
import datetime
import random
import sys

import numpy as np

from reader import ReaderCupt, fileCompletelyRead, isInASequence

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


parser = argparse.ArgumentParser(description="""
        System to train/test recognition of multi word expressions.
        """)
parser.add_argument("-feat", "--featureColumns", dest="featureColumns",
                    required=False, nargs='+', type=int, default=[3, 4],
                    help="""
                    To treat columns as features. The first column is number 1, the second 2...
                    By default, features are LEMME and POS, e.g 3 4 
                    """)
parser.add_argument("--mweTags", type=int,
                    required=False, dest='mweTags', default=11,
                    help="""
                    To give the number of the column containing tags (default 11)
                    Careful! The first column is number 1, the second number 2, ...
                    """)
parser.add_argument("--embeddings", nargs='+', type=str, dest="embeddingsArgument",
                    help="""
                    To give some files containing embeddings.
                    First, you give the path of the file containing embeddings,
                    and separate with a \",\" you gave the column concern by this file.
                    eg: file1,2 file2,5
                    Careful! You could have only column match with featureColumns.
                    """)
parser.add_argument("--file", metavar="filename", dest="filename", required=True, type=argparse.FileType('r'),
                    help="""
                    Give a file in the Extended CoNLL-U (.cupt) format.
                    You can only give one file to train/test a model.
                    You can give a CoNLL file to only test it.
                    """)
parser.add_argument("--mode", type=str, dest='mode', required=True,
                    help="""
                    To choice the mode of the system : train/test.
                    If the file is a train file and you want to create a model use \'train\'.
                    If the file is a test/dev file and you want to load a model use \'test\'.
                    In test mode the system doesn't need params RNN.
                    """)
parser.add_argument("--model", action='store', type=str,
                    required=True, dest='model',
                    help="""
                    Name of the model which you want to save/load without extension.
                    e.g \'nameModel\' , and the system save/load files nameModel.h5, nameModel.voc and nameModel.args.
                    nameModel.h5 is the model file.
                    nameModel.voc is the vocabulary file.
                    nameModel.args is the arguments file which train your model.
                    """)
parser.add_argument("--io", action='store_const', const=True,
                    dest='io',
                    help="""
                    Option to use the representation of IO.
                    You can combine with other options like --nogap or/and --cat.
                    By default, the representation is BIO.
                    """)
parser.add_argument("-ng", "--nogap", action='store_const', const=True,
                    dest='nogap',
                    help="""
                    Option to use the representation of BIO/IO without gap.
                    By default, the gap it is using to the representation of BIO/IO.
                    """)
parser.add_argument("-cat", "--category", action='store_const', const=True,
                    dest='withMWE',
                    help="""
                    Option to use the representation of BIO/IO with categories.
                    By default, the representation of BIO/IO is without categories.
                    """)
parser.add_argument("--sentences_per_batch", required=False, type=int,
                    dest='batch_size', default=128,
                    help="""
                    Option to initialize the sentences numbers for batch to train RNN.
                    By default, sentences_per_batch is 128.
                    """)
parser.add_argument("--max_sentence_size", required=False, type=int,
                    dest='max_sentence_size', default=128,
                    help="""
                    Option to initialize the size of sentence for the RNN.
                    By default, max_sentence_size is 128.
                    """)
parser.add_argument("--overlaps", action='store_const', const=True, dest='withOverlaps',
                    help="""
                    Option to use the representation of BIO/IO with overlaps.
                    We can't load a file test with overlaps, if option test and overlaps are activated, only the option test is considered.
                    By default, the representation is without overlaps. 
                    """)
parser.add_argument("--validation_split", required=False, type=float,
                    dest='validation_split', default=0.0,
                    help="""
                    Option to configure the validation data to train the RNN.
                    By default 0.0 of train file is use to validation data.
                    """)
parser.add_argument("--validation_data", required=False, metavar="validation_data", dest="validation_data",
                    type=argparse.FileType('r'),
                    help="""
                    Give a file in the Extended CoNLL-U (.cupt) format to loss function for the RNN.
                    """)
parser.add_argument("--epochs", required=False, metavar="epoch", dest="epoch", type=int, default=10,
                    help="""
                    Number of epochs to train RNN.
                    By default, RNN trains on 10 epochs.
                    """)
parser.add_argument("--recurrent_unit", required=False, metavar="recurrent_unit", dest="recurrent_unit", type=str,
                    default="biGRU",
                    help="""
                    This option allows choosing the type of recurrent units in the recurrent layer. By default it is biGRU.
                    You can choice GRU, LSTM, biGRU, biLSTM.
                    """)
parser.add_argument("--number_recurrent_layer", required=False, metavar="number_recurrent_layer",
                    dest="number_recurrent_layer", type=int, default=2,
                    help="""
                    This option allows choosing the numbers of recurrent layer. By default it is 2 recurrent layers.
                    """)
parser.add_argument("--size_recurrent_layer", required=False, metavar="size_recurrent_layer",
                    dest="size_recurrent_layer", type=int, default=512,
                    help="""
                    This option allows choosing the size of recurrent layer. By default it is 512.
                    """)
parser.add_argument("--feat_embedding_size", required=False, metavar="feat_embedding_size", dest="feat_embedding_size",
                    type=int, default=[128, 64], nargs='+',
                    help="""
                    Option that takes as input a sequence of integers corresponding to the dimension/size of the embeddings layer of each column given to the --feat option.
                    By default, all embeddings have the same size, use the current default value (64)
                    """)
parser.add_argument("--early_stopping_mode", required=False, metavar="early_stopping_mode", dest="early_stopping_mode",
                    type=str, default="loss",
                    help="""
                    Option to save the best model training in function of acc/loss value, only if you use validation_data or validation_split.
                    By default, it is in function of the loss value.
                    """)
parser.add_argument("--patience_early_stopping", required=False, metavar="patience_early_stopping",
                    dest="patience_early_stopping", type=int, default=5,
                    help="""
                    Option to choice patience for the early stopping.
                    By default, it is 5 epochs.
                    """)
parser.add_argument("--numpy_seed", required=False, metavar="numpy_seed", dest="numpy_seed", type=int,
                    default=42,
                    help="""
                    Option to initialize manually the seed of numpy.
                    By default, it is initialized to 42.
                    """)
parser.add_argument("--tensorflow_seed", required=False, metavar="tensorflow_seed", dest="tensorflow_seed", type=int,
                    default=42,
                    help="""
                    Option to initialize manually the seed of tensorflow.
                    By default, it is initialized to 42.
                    """)
parser.add_argument("--random_seed", required=False, metavar="random_seed", dest="random_seed", type=int,
                    default=42,
                    help="""
                    Option to initialize manually the seed of random library.
                    By default, it is initialized to 42.
                    """)
parser.add_argument("--dropout", required=False, metavar="dropout", dest="dropout", type=float,
                    default=0.0,
                    help="""
                    Float between 0 and 1.
                    Fraction of the units to drop for the linear transformation of the inputs.
                    """)
parser.add_argument("--recurrent_dropout", required=False, metavar="recurrent_dropout", dest="recurrent_dropout",
                    type=float,
                    default=0.0,
                    help="""
                    Float between 0 and 1.
                    Fraction of the units to drop for the linear transformation of the recurrent state.
                    """)

# TODO / legend
parser.add_argument("--no_fine_tuning_embeddings", required=False, metavar="no_fine_tune_embeddings",
                    dest="no_fine_tune_embeddings", const=True, nargs='?',
                    help="""
                    Option to no tune embeddings in train.
                    We can't used its option without --embeddings.
                    """)
parser.add_argument("--activationCRF", required=False, metavar="activationCRF",
                    dest="activationCRF", const=True, nargs='?',
                    help="""
                    Option to replace activation('softmax') by a CRF layer.
                    """)

numColTag = 0
colIgnore = []
embeddingsArgument = dict()
nbFeat = 0
codeInterestingTags = []
filename = None
isTrain = False
isTest = False
filenameModelWithoutExtension = None
FORMAT = None
recurrent_unit = None
number_recurrent_layer = None
monitor = None
monitor_mode = None
patience = None
dropout = None
recurrent_dropout = None
trainable_embeddings = None
activationCRF = None


def uniq(seq):
    # not order preserving
    set = {}
    map(set.__setitem__, seq, [])
    return set.keys()


def treat_options(args):
    global isTrain
    global isTest

    if args.mode.lower() == "train":
        isTrain = True
        isTest = False
    elif args.mode.lower() == "test":
        isTrain = False
        isTest = True
    else:
        sys.stderr.write("Error with argument --mode (train/test)\n")
        exit(-10)

    global filename
    global filenameModelWithoutExtension

    filename = args.filename
    filenameModelWithoutExtension = args.model

    if isTrain:

        global numColTag
        global embeddingsArgument

        global FORMAT
        global recurrent_unit
        global number_recurrent_layer
        global monitor
        global monitor_mode
        global patience
        global dropout
        global recurrent_dropout
        global trainable_embeddings
        global activationCRF

        if args.io:
            FORMAT = "IO"
        else:
            FORMAT = "BIO"

        if not args.nogap:
            FORMAT += "g"

        if args.withMWE:
            FORMAT += "cat"

        numColTag = args.mweTags - 1

        number_recurrent_layer = args.number_recurrent_layer
        patience = args.patience_early_stopping
        dropout = args.dropout
        recurrent_dropout = args.recurrent_dropout
        if args.activationCRF:
            activationCRF = args.activationCRF
        else:
            activationCRF = False
            args.activationCRF = False

        if args.embeddingsArgument:
            embeddingsFileAndCol = args.embeddingsArgument
            for i in range(len(embeddingsFileAndCol)):
                embeddingsFileAndCol[i] = embeddingsFileAndCol[i].split(",")
                fileEmbed = embeddingsFileAndCol[i][0]
                numCol = int(embeddingsFileAndCol[i][1]) - 1
                if embeddingsArgument.has_key(int(numCol)):
                    sys.stderr.write("Error with argument --embeddings")
                    exit()
                embeddingsArgument[int(numCol)] = fileEmbed

        if args.recurrent_unit.lower() not in ["gru", "lstm", "bigru", "bilstm"]:
            sys.stderr.write("Error with the argument --recurrent_unit.\n")
            exit(40)
        else:
            recurrent_unit = args.recurrent_unit.lower()

        if len(args.feat_embedding_size) != 1 and len(args.feat_embedding_size) != len(args.featureColumns):
            sys.stderr.write("Error with argument --feat_embedding_size\n")
            exit(41)

        for embed in args.feat_embedding_size:
            if embed < 1:
                sys.stderr.write("Error with argument --feat_embedding_size, size < 1\n")
                exit(41)

        if args.no_fine_tune_embeddings:
            trainable_embeddings = False
        elif not args.no_fine_tune_embeddings:
            trainable_embeddings = True
        elif args.no_fine_tune_embeddings and not args.embeddings:
            sys.stderr.write("Error : You can't use --no_fine_tune_embeddings without give --embeddings")
            exit(300)

        if args.early_stopping_mode.lower() == "acc":
            monitor = "val_acc"
            monitor_mode = "max"
        elif args.early_stopping_mode.lower() == "loss":
            monitor = "val_loss"
            monitor_mode = "min"

        save_args(filenameModelWithoutExtension + ".args", args)
    else:

        load_args(filenameModelWithoutExtension + ".args", args)


def enumdict():
    a = collections.defaultdict(lambda: len(a))
    return a


def init(line, features, vocab):
    global nbFeat
    curSequence = []
    for feati in range(nbFeat):
        if (feati == numColTag):
            tagsCurSeq = []
        curSequence.append([])
        features.append([])
        vocab[feati]["<unk>"] = 1
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


def load_text_train(filename, vocab):
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


def load_text_test(filename, vocab):
    global nbFeat
    start = True
    features = []
    tags = []
    for sentence in filename:
        for line in sentence:
            if (nbFeat == 0):
                nbFeat = len(line.strip().split("\t"))
            if (start == True):
                curSequence, features, vocab, tagsCurSeq = init(line, features, vocab)
                start = False
            if (line == "\n"):
                tags, tagsCurSeq, features, curSequence = handleEndOfSequence(tags, tagsCurSeq, features, curSequence)
            else:
                tagsCurSeq, vocab, curSequence = handleSequence(line, tagsCurSeq, vocab, curSequence)

    return features, tags, vocab


r"""
    Save arguments which train model
"""


def save_args(nameFileArgs, args):
    file = open(nameFileArgs, "w")

    global FORMAT
    file.write("FORMAT" + "\t" + FORMAT + "\n")
    file.write("mweTags" + "\t" + str(args.mweTags) + "\n")
    file.write("batch_size" + "\t" + str(args.batch_size) + "\n")
    file.write("max_sentence_size" + "\t" + str(args.max_sentence_size) + "\n")
    file.write("recurrent_unit" + "\t" + str(args.recurrent_unit) + "\n")
    file.write("number_recurrent_layer" + "\t" + str(args.number_recurrent_layer) + "\n")
    file.write("size_recurrent_layer" + "\t" + str(args.size_recurrent_layer) + "\n")
    file.write("numpy_seed" + "\t" + str(args.numpy_seed) + "\n")
    file.write("tensorflow_seed" + "\t" + str(args.tensorflow_seed) + "\n")
    file.write("random_seed" + "\t" + str(args.random_seed) + "\n")
    file.write("activationCRF" + "\t" + str(args.activationCRF) + "\n")
    file.write("feat_embedding_size" + "\t")
    for col in args.feat_embedding_size:
        file.write(str(col) + " ")
    file.write("\n")

    file.write("featureColumns" + "\t")
    for col in args.featureColumns:
        file.write(str(col) + " ")
    file.write("\n")
    global embeddingsArgument
    file.write("embeddings" + "\t")

    for key in embeddingsArgument:
        file.write(str(embeddingsArgument[key]) + "," + str(key) + " ")
    file.write("\n")


r"""
    Load arguments to test the model
"""


def load_args(nameFileArgs, args):
    global FORMAT
    global numColTag
    global recurrent_unit
    global number_recurrent_layer
    global activationCRF

    dictArgs = {}
    with open(nameFileArgs) as fa:
        for line in fa:
            dictArgs[line.split("\t")[0]] = line.split("\t")[1]

    FORMAT = dictArgs.get("FORMAT").split("\n")[0]
    args.mweTags = int(dictArgs.get("mweTags").split("\n")[0])
    numColTag = args.mweTags - 1
    args.batch_size = int(dictArgs.get("batch_size").split("\n")[0])
    args.max_sentence_size = int(dictArgs.get("max_sentence_size").split("\n")[0])
    recurrent_unit = dictArgs.get("recurrent_unit").split("\n")[0].lower()
    number_recurrent_layer = int(dictArgs.get("number_recurrent_layer").split("\n")[0])
    args.size_recurrent_layer = int(dictArgs.get("size_recurrent_layer").split("\n")[0])
    args.numpy_seed = int(dictArgs.get("numpy_seed").split("\n")[0])
    args.tensorflow_seed = int(dictArgs.get("tensorflow_seed").split("\n")[0])
    args.random_seed = int(dictArgs.get("random_seed").split("\n")[0])
    activationCRF = dictArgs.get("activationCRF").split("\n")[0]

    if 'False' == activationCRF:
        activationCRF = False
    elif activationCRF == 'True':
        activationCRF = True

    args.feat_embedding_size = []

    for feat in dictArgs.get("feat_embedding_size").split("\n")[0].split(" "):
        if feat != '':
            args.feat_embedding_size.append(int(feat))

    args.featureColumns = []

    for feat in dictArgs.get("featureColumns").split("\n")[0].split(" "):
        if feat != '':
            args.featureColumns.append(int(feat))

    global embeddingsArgument

    for feat in dictArgs.get("embeddings").split("\n")[0].split(" "):
        if feat != '':
            embeddingsArgument[int(feat.split(",")[1])] = feat.split(",")[0]


r"""
    Load and return the vocabulary dict.
"""


def loadVocab(nameFileVocab):
    vocab = collections.defaultdict(enumdict)
    index = 0
    vocab[index] = collections.defaultdict(enumdict)
    with open(nameFileVocab) as fv:
        for line in fv:
            if fileCompletelyRead(line):
                pass
            elif isInASequence(line):
                feat = str(line.split("\t")[0])
                ind = int(line.split("\t")[1])
                vocab[index][feat] = ind
            else:
                index += 1
                vocab[index] = collections.defaultdict(enumdict)

    return vocab


def vectorize(features, tags, vocab, unroll):
    X_train = []
    from keras.preprocessing.sequence import pad_sequences

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
            if curTag in codeInterestingTags:
                sample_weight[i][j] = 1.0
            else:
                sample_weight[i][j] = 0.01
            if Y_train[i, j, 0] != 0:
                mask[i, j, 0] = 1

    for i in colIgnore:
        X_train.pop(i)

    return X_train, Y_train, mask, sample_weight


def make_model_gru(hidden, embeddings, num_tags, inputs):
    import keras
    from keras.models import Model
    from keras.layers import GRU, Dense, Activation, TimeDistributed
    global number_recurrent_layer
    global dropout
    global recurrent_dropout
    global activationCRF

    x = keras.layers.concatenate(embeddings)
    for recurrent_layer in range(number_recurrent_layer):
        x = GRU(hidden, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)(x)
        x = TimeDistributed(Dense(num_tags))(x)
    if activationCRF:
        from keras_contrib.layers import CRF
        crf = CRF(num_tags, sparse_target=True)
        x = crf(x)
        model = Model(inputs=inputs, outputs=[x])
        model.compile(loss=crf.loss_function, optimizer='Nadam', metrics=[crf.accuracy])
    else:
        x = Activation('softmax')(x)
        model = Model(inputs=inputs, outputs=[x])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['acc'],
                      sample_weight_mode="temporal")  ###############################
    return model


def make_model_bigru(hidden, embeddings, num_tags, inputs):
    import keras
    from keras.models import Model
    from keras.layers import GRU, Dense, Activation, TimeDistributed, Bidirectional
    global number_recurrent_layer
    global dropout
    global recurrent_dropout
    global activationCRF

    x = keras.layers.concatenate(embeddings)
    for recurrent_layer in range(number_recurrent_layer):
        x = Bidirectional(GRU(hidden, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout))(x)
        x = TimeDistributed(Dense(num_tags))(x)
    if activationCRF:
        from keras_contrib.layers import CRF
        crf = CRF(num_tags, sparse_target=True)
        x = crf(x)
        model = Model(inputs=inputs, outputs=[x])
        model.compile(loss=crf.loss_function, optimizer='Nadam', metrics=[crf.accuracy])
    else:
        x = Activation('softmax')(x)
        model = Model(inputs=inputs, outputs=[x])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['acc'],
                      sample_weight_mode="temporal")
    return model


def make_model_lstm(hidden, embeddings, num_tags, inputs):
    import keras
    from keras.models import Model
    from keras.layers import GRU, Dense, Activation, TimeDistributed
    global number_recurrent_layer
    global dropout
    global recurrent_dropout
    global activationCRF

    x = keras.layers.concatenate(embeddings)
    for recurrent_layer in range(number_recurrent_layer):
        x = GRU(hidden, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)(x)
        x = TimeDistributed(Dense(num_tags))(x)
    if activationCRF:
        from keras_contrib.layers import CRF
        crf = CRF(num_tags, sparse_target=True)
        x = crf(x)
        model = Model(inputs=inputs, outputs=[x])
        model.compile(loss=crf.loss_function, optimizer='Nadam', metrics=[crf.accuracy])
    else:
        x = Activation('softmax')(x)
        model = Model(inputs=inputs, outputs=[x])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['acc'],
                      sample_weight_mode="temporal")  ###############################
    return model


def make_model_bilstm(hidden, embeddings, num_tags, inputs):
    import keras
    from keras.models import Model
    from keras.layers import LSTM, Dense, Activation, TimeDistributed, Bidirectional
    global number_recurrent_layer
    global dropout
    global recurrent_dropout
    global activationCRF

    x = keras.layers.concatenate(embeddings)
    for recurrent_layer in range(number_recurrent_layer):
        x = Bidirectional(LSTM(hidden, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout))(x)
        x = TimeDistributed(Dense(num_tags))(x)
    if activationCRF:
        from keras_contrib.layers import CRF
        crf = CRF(num_tags, sparse_target=True)
        x = crf(x)
        model = Model(inputs=inputs, outputs=[x])
        model.compile(loss=crf.loss_function, optimizer='Nadam', metrics=[crf.accuracy])
    else:
        x = Activation('softmax')(x)
        model = Model(inputs=inputs, outputs=[x])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['acc'],
                      sample_weight_mode="temporal")
    return model


def make_modelMWE(hidden, embed, num_tags, unroll, vocab):
    from keras.layers import Embedding, Input
    global recurrent_unit
    global trainable_embeddings
    global nbFeat
    inputs = []
    embeddings = []

    if isTest:
        nbFeat = len(vocab) - 1

    for i in range(nbFeat):
        if i in colIgnore:
            continue
        nameInputFeat = 'Column' + str(i)
        inputFeat = Input(shape=(unroll,), dtype='int32', name=nameInputFeat)
        inputs.append(inputFeat)
        if i in embeddingsArgument:
            embedding_matrix, vocab, dimension = loadEmbeddings(vocab, embeddingsArgument[i], i)
            x = Embedding(output_dim=dimension, input_dim=len(vocab[i]), weights=[embedding_matrix],
                          input_length=unroll, trainable=trainable_embeddings)(inputFeat)
        else:
            x = Embedding(output_dim=embed.get(i), input_dim=len(vocab[i]), input_length=unroll,
                          trainable=trainable_embeddings)(inputFeat)
        embeddings.append(x)

    if recurrent_unit == "gru":
        model = make_model_gru(hidden, embeddings, num_tags, inputs)
    elif recurrent_unit == "bigru":
        model = make_model_bigru(hidden, embeddings, num_tags, inputs)
    elif recurrent_unit == "lstm":
        model = make_model_lstm(hidden, embeddings, num_tags, inputs)
    elif recurrent_unit == "bilstm":
        model = make_model_bilstm(hidden, embeddings, num_tags, inputs)
    else:
        sys.stderr.write("Error recurrent unit\n")
        exit(400)

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
                prediction[i][j][0] = maxTag

    return prediction


def genereTag(prediction, vocab, unroll):
    rev_vocabTags = {i: char for char, i in vocab[numColTag].items()}
    # sys.stderr.write(str(rev_vocabTags) + "\n")
    pred = []
    listNbToken = []
    for i in range(len(prediction)):
        tag = []
        nbToken = 0
        for j in range(unroll - 1):
            curTagEncode = prediction[i][j][0]
            if curTagEncode == 0:
                break
            else:
                #    sys.stderr.write(str(rev_vocabTags[curTagEncode]))
                tag.append(rev_vocabTags[curTagEncode])
                nbToken += 1
            # sys.stderr.write(rev_vocabTags[curTagEncode] + "\n")
        pred.append(tag)
        # sys.stderr.write("\n")
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

    hidden = args.size_recurrent_layer
    batch = args.batch_size
    unroll = args.max_sentence_size
    validation_split = args.validation_split
    validation_data = args.validation_data
    embed = {}

    for index in range(len(args.featureColumns)):
        embed[int(args.featureColumns[index]) - 1] = int(
            args.feat_embedding_size[index % len(args.feat_embedding_size)])
    epochs = args.epoch
    vocab = []

    sys.stderr.write("Load FORMAT ..\t")
    print(str(datetime.datetime.now()), file=sys.stderr)
    reformatFile = ReaderCupt(FORMAT, args.withOverlaps, isTest, filename, numColTag)
    reformatFile.run()

    global colIgnore

    colIgnore = list(range(reformatFile.numberOfColumns))
    for index in args.featureColumns:
        colIgnore.remove(index - 1)
    # colIgnore = uniq(colIgnore)
    colIgnore.sort(reverse=True)

    if validation_data is not None:
        devFile = ReaderCupt(FORMAT, False, True, validation_data, numColTag)
        devFile.run()

    os.environ['PYTHONHASHSEED'] = '0'
    from numpy.random import seed
    seed(args.numpy_seed)

    import tensorflow as tf
    tf.set_random_seed(args.tensorflow_seed)

    random.seed(args.random_seed)

    from keras import backend as K

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1,
                                  )
    # Force Tensorflow to use a single thread
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

    K.set_session(sess)

    sys.stderr.write("Env session keras : numpy_seed(" + str(args.numpy_seed) + "), tensorflow_seed(" + str(
        args.tensorflow_seed) + "), random_seed(" + str(args.random_seed) + ")..\n")

    if isTrain:

        sys.stderr.write("Load training file..\n")
        features, tags, vocab = load_text_train(reformatFile.resultSequences, vocab)
        X, Y, mask, sample_weight = vectorize(features, tags, vocab, unroll)

        num_tags = len(vocab[numColTag])

        sys.stderr.write("Create model..\n")
        model = make_modelMWE(hidden, embed, num_tags, unroll, vocab)
        # plot_model(model, to_file='modelMWE.png', show_shapes=True)
        from keras.callbacks import ModelCheckpoint, EarlyStopping

        if validation_data is None:

            if validation_split > 0:
                sys.stderr.write("Starting training with validation_split...\n")
                checkpoint = ModelCheckpoint(filenameModelWithoutExtension + '.h5', monitor=monitor, verbose=1,
                                             save_best_only=True,
                                             mode=monitor_mode)

                earlyStopping = EarlyStopping(monitor=monitor, patience=patience, verbose=1, mode=monitor_mode)
                callbacks_list = [checkpoint, earlyStopping]

                if activationCRF:
                    model.fit(X, Y, batch_size=batch, epochs=epochs, shuffle=True,
                              validation_split=validation_split, callbacks=callbacks_list)
                else:
                    model.fit(X, Y, batch_size=batch, epochs=epochs, shuffle=True,
                              sample_weight=sample_weight, validation_split=validation_split, callbacks=callbacks_list)
            else:
                sys.stderr.write("Starting training without validation_split...\n")

                if activationCRF:
                    model.fit(X, Y, batch_size=batch, epochs=epochs, shuffle=True, validation_split=validation_split)
                else:
                    model.fit(X, Y, batch_size=batch, epochs=epochs, shuffle=True,
                              validation_split=validation_split, sample_weight=sample_weight)

                sys.stderr.write("Save model\n")
                model.save(filenameModelWithoutExtension + '.h5')

        else:

            sys.stderr.write("Load dev file..\n")
            devFile.verifyUnknowWord(vocab)
            features, tags, useless = load_text_test(devFile.resultSequences, vocab)

            X_test, Y_test, mask, useless = vectorize(features, tags, vocab, unroll)

            sys.stderr.write("Starting training with validation_data ...\n")
            checkpoint = ModelCheckpoint(filenameModelWithoutExtension + '.h5', monitor=monitor, verbose=1,
                                         save_best_only=True,
                                         mode=monitor_mode)
            earlyStopping = EarlyStopping(monitor=monitor, patience=patience, verbose=1, mode=monitor_mode,
                                          )
            callbacks_list = [checkpoint, earlyStopping]
            if activationCRF:
                model.fit(X, Y, batch_size=batch, epochs=epochs, shuffle=True,
                          validation_data=(X_test, Y_test), callbacks=callbacks_list)
            else:
                model.fit(X, Y, batch_size=batch, epochs=epochs, shuffle=True,
                          validation_data=(X_test, Y_test), sample_weight=sample_weight, callbacks=callbacks_list)

        sys.stderr.write("Save vocabulary...\n")

        reformatFile.saveVocab(filenameModelWithoutExtension + '.voc', vocab)

        sys.stderr.write("END training\t")
        print(str(datetime.datetime.now()) + "\n", file=sys.stderr)

    elif isTest:

        sys.stderr.write("Load vocabulary...\n")
        vocab = loadVocab(filenameModelWithoutExtension + ".voc")
        reformatFile.verifyUnknowWord(vocab)
        sys.stderr.write("Load model..\n")
        from keras.models import load_model

        if activationCRF:
            from keras_contrib.utils import save_load_utils
            num_tags = len(vocab[args.mweTags - 1])
            model = make_modelMWE(hidden, embed, num_tags, unroll, vocab)
            save_load_utils.load_all_weights(model, filenameModelWithoutExtension + '.h5', include_optimizer=False)
        else:
            model = load_model(filenameModelWithoutExtension + '.h5')

        # model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['acc'],
        #              sample_weight_mode="temporal")
        sys.stderr.write("Load testing file..\n")

        features, tags, useless = load_text_test(reformatFile.resultSequences, vocab)
        X, Y, mask, useless = vectorize(features, tags, vocab, unroll)

        classes = model.predict(X)
        # sys.stderr.write(classes.shape+ "\nclasses: "+ classes)

        prediction = maxClasses(classes, Y, unroll, mask)

        sys.stderr.write("Add prediction...\n")
        pred, listNbToken = genereTag(prediction, vocab, unroll)
        reformatFile.addPrediction(pred, listNbToken)
        reformatFile.printFileCupt()

        # print(len(pred))
        sys.stderr.write("END testing\t")
        print(str(datetime.datetime.now()), file=sys.stderr)

    else:
        sys.stderr("Error argument: Do you want to test or train ?")
        exit(-2)


main()
