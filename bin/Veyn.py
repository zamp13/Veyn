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
from fasttext_preprocessing import PreprocessingFasttext
from w2v_preprocessing import PreprocessingW2V
import os
from tqdm import tqdm

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
parser.add_argument("--file", metavar="filename", dest="filename", required=True,
                    type=argparse.FileType('r', encoding="utf8"),
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
                    type=argparse.FileType('r', encoding="utf-8"),
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
                    type=int, default=[128, 20], nargs='+',
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
parser.add_argument("--save_cupt", required=False, metavar="file_save_cupt", dest="file_save_cupt",
                    type=argparse.FileType('w', encoding="utf-8"),
                    help="""
                    Give a file to save the prediction at format cupt. (Only to test model)
                    """)
parser.add_argument("-conv", "--convolution", required=False, metavar="convolution_layer",
                    dest="convolution_layer", const=True, nargs='?',
                    help="""
                    Option to add convolution layer before recurrent layer to extract n_gram.
                    Only uses option with activationCRF pls, otherwise it do not works 
                    """)
parser.add_argument("--noreplace", required=False, metavar="noreplace",
                    dest="noreplace", const=True, nargs='?',
                    help="""
                    Option to no replace the unknow word when you use fasttext representation.
                    """)
parser.add_argument("--min_count", required=False, metavar="min_count", dest="min_count",
                    type=int,
                    default=[1, 1], nargs='+',
                    help="""
                    Integer >= 1.
                    Option to choose the min_count in train corpus to train an embedding for unknown word.
                    If word occurrences is > min_count, the word is replaced by <unk>.
                    The number of arguments do the same as the number of featureColumns arguments !
                    """)
# Fasttext option
parser.add_argument("--fasttext", required=False, metavar="fasttext",
                    dest="fasttext", type=str, nargs='+',
                    help="""
                    Option to add fasttext pretrained embeddings.
                    This option use a list of <name_model,column_feature,train/load>
                    Watch other options which begining by "--fasttext" to param fasttext model.
                    - train: trained a new fasttext model.
                    - load: load a fasttext model with patch = name_model
                    If you have more 2 model, you need to add other fasttext's options to run.
                    Ex: model1,2,load model2,3,train ,etc...
                    """)
parser.add_argument("--fasttext_size", required=False, metavar="fasttext_size",
                    dest="fasttext_size", type=int, nargs='+', default=[128],
                    help="""
                    Option to select the dimensional of embeddings.
                    This option use a list of <size>. By default size = 128.
                    """)
parser.add_argument("--fasttext_window", required=False, metavar="fasttext_window",
                    dest="fasttext_window", type=int, nargs='+', default=[5],
                    help="""
                    Option to select the window of fasttext model.
                    This option use a list of <window>. By default window = 5.
                    """)
parser.add_argument("--fasttext_epochs", required=False, metavar="fasttext_epochs",
                    dest="fasttext_epochs", type=int, nargs='+', default=[10],
                    help="""
                    Option to select the number of epochs to train fasttext model.
                    This option use a list of <epochs>. By default epochs = 10.
                    """)
parser.add_argument("--fasttext_min_count", required=False, metavar="fasttext_min_count",
                    dest="fasttext_min_count", type=int, nargs='+', default=[1],
                    help="""
                    Option to select the min_count to train fasttext model.
                    This option use a list of <min_count>. By default epochs = 1.
                    """)
parser.add_argument("--fasttext_word_ngram", required=False, metavar="fasttext_word_ngram",
                    dest="fasttext_word_ngram", type=int, nargs='+', default=[1],
                    help="""
                    Option to select the min_count to train fasttext model.
                    This option use a list of <min_count>. By default epochs = 1.
                    """)
parser.add_argument("--fasttext_save_w2v_format", required=False, metavar="fasttext_save_w2v_format",
                    dest="fasttext_save_w2v_format", type=str, nargs='+',
                    help="""
                    Option to select save the embeddings train at the format w2v.
                    This option use a list of <name_file> different to the name of model.
                    """)
# Word2Vec option
parser.add_argument("--w2v", required=False, metavar="w2v",
                    dest="w2v", type=str, nargs='+',
                    help="""
                    Option to add w2v pretrained embeddings.
                    This option use a list of <name_model,column_feature,train/load>
                    Watch other options which begining by "--w2v" to param w2v model.
                    - train: trained a new w2v model.
                    - load: load a w2v model with patch = name_model
                    If you have more 2 model, you need to add other w2v's options to run.
                    Ex: model1,2,load model2,3,train ,etc...
                    """)
parser.add_argument("--w2v_size", required=False, metavar="w2v_size",
                    dest="w2v_size", type=int, nargs='+', default=[128],
                    help="""
                    Option to select the dimensional of embeddings.
                    This option use a list of <size>. By default size = 128.
                    """)
parser.add_argument("--w2v_window", required=False, metavar="w2v_window",
                    dest="w2v_window", type=int, nargs='+', default=[5],
                    help="""
                    Option to select the window of w2v model.
                    This option use a list of <window>. By default window = 5.
                    """)
parser.add_argument("--w2v_epochs", required=False, metavar="w2v_epochs",
                    dest="w2v_epochs", type=int, nargs='+', default=[10],
                    help="""
                    Option to select the number of epochs to train w2v model.
                    This option use a list of <epochs>. By default epochs = 10.
                    """)
parser.add_argument("--w2v_min_count", required=False, metavar="w2v_min_count",
                    dest="w2v_min_count", type=int, nargs='+', default=[1],
                    help="""
                    Option to select the min_count to train w2v model.
                    This option use a list of <min_count>. By default epochs = 1.
                    """)

# TODO ~ AJOUTER le status de fasttext

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
convolution_layer = None
fasttexts_model = {}
w2v_model = {}
noreplace = False
min_count = {}


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
        global convolution_layer
        global fasttexts_model
        global w2v_model
        global noreplace
        global min_count

        if args.io:
            FORMAT = "IO"
        else:
            FORMAT = "BIO"

        if not args.nogap:
            FORMAT += "g"

        if args.withMWE:
            FORMAT += "cat"

        if len(args.featureColumns) == len(args.min_count):
            for i in range(len(args.featureColumns)):
                min_count[args.featureColumns[i] - 1] = args.min_count[i]
        else:
            print("Warning: option min_count: min_count have not the same arguments number than featureColumns",
                  file=sys.stderr)
            for i in range(len(args.featureColumns)):
                min_count[args.featureColumns[i] - 1] = 1

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

        if args.convolution_layer:
            convolution_layer = args.convolution_layer
        else:
            convolution_layer = False
            args.convolution_layer = False

        if args.noreplace:
            noreplace = args.noreplace

        if args.embeddingsArgument:
            embeddingsFileAndCol = args.embeddingsArgument
            for i in range(len(embeddingsFileAndCol)):
                embeddingsFileAndCol[i] = embeddingsFileAndCol[i].split(",")
                fileEmbed = embeddingsFileAndCol[i][0]
                numCol = int(embeddingsFileAndCol[i][1]) - 1
                if numCol in embeddingsArgument:
                    sys.stderr.write("Error with argument --embeddings")
                    exit()
                embeddingsArgument[numCol] = fileEmbed

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

        if args.fasttext:

            if verify_fasttext_argument(
                    [args.fasttext, args.fasttext_size, args.fasttext_word_ngram, args.fasttext_epochs,
                     args.fasttext_window]):

                for i in range(len(args.fasttext)):
                    f = args.fasttext[i].split(",")
                    name_model = f[0]
                    feat = int(f[1]) - 1
                    train = f[2]
                    if train.lower() == "train":
                        train = True
                    elif train.lower() == "load":
                        train = False
                    min_count[feat] = args.fasttext_min_count[i]
                    fasttexts_model[feat] = PreprocessingFasttext(name_model, train=train, size=args.fasttext_size[i],
                                                                  window=args.fasttext_window[i],
                                                                  word_ngram=args.fasttext_word_ngram[i],
                                                                  min_count=args.fasttext_min_count[i],
                                                                  epochs=args.fasttext_epochs[i])
            else:
                print("Error : arguments for model fasttext.", file=sys.stderr)
        if args.w2v:

            if verify_w2v_argument(
                    [args.w2v, args.w2v_size, args.w2v_epochs, args.w2v_window]):

                for i in range(len(args.w2v)):
                    f = args.w2v[i].split(",")
                    name_model = f[0] + ".bin"
                    feat = int(f[1]) - 1
                    train = f[2]
                    if train.lower() == "train":
                        train = True
                    elif train.lower() == "load":
                        train = False
                    min_count[feat] = args.w2v_min_count[i]
                    w2v_model[feat] = PreprocessingW2V(name_model, train=train, size=args.w2v_size[i],
                                                       window=args.w2v_window[i],
                                                       min_count=args.w2v_min_count[i],
                                                       epochs=args.w2v_epochs[i])
            else:
                print("Error : arguments for model w2v.", file=sys.stderr)
        save_args(filenameModelWithoutExtension + ".args", args)
    else:

        load_args(filenameModelWithoutExtension + ".args", args)
        if args.fasttext:
            for i in range(len(args.fasttext)):
                f = args.fasttext[i].split(",")
                name_model = f[0]
                feat = int(f[1]) - 1
                train = f[2]
                if train.lower() == "load":
                    train = False
                else:
                    print("Error : arguments for model fasttext. You can't train a new model on test.")
                fasttexts_model[feat] = PreprocessingFasttext(name_model, train=train)
        if args.w2v:
            for i in range(len(args.w2v)):
                f = args.w2v[i].split(",")
                name_model = f[0] + ".bin"
                feat = int(f[1]) - 1
                train = f[2]
                if train.lower() == "train":
                    train = True
                elif train.lower() == "load":
                    train = False
                w2v_model[feat] = PreprocessingW2V(name_model, train=train)


def verify_fasttext_argument(option_fasttext):
    for index_option in range(len(option_fasttext)):
        for index2_option in range(len(option_fasttext)):
            if index_option != index2_option:
                if len(option_fasttext[index_option]) != len(option_fasttext[index2_option]):
                    return False
    return True


def verify_w2v_argument(option_w2v):
    for index_option in range(len(option_w2v)):
        for index2_option in range(len(option_w2v)):
            if index_option != index2_option:
                if len(option_w2v[index_option]) != len(option_w2v[index2_option]):
                    return False
    return True


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
        vocab[feati]["<pad>"] = 0
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
    file.write("convolution_layer" + "\t" + str(args.convolution_layer) + "\n")
    file.write("dropout" + "\t" + str(args.dropout) + "\n")
    file.write("recurrent_dropout" + "\t" + str(args.recurrent_dropout) + "\n")
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
    global convolution_layer
    global dropout
    global recurrent_dropout

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
    convolution_layer = dictArgs.get("convolution_layer").split("\n")[0]
    dropout = float(dictArgs.get("dropout").split("\n")[0])
    recurrent_dropout = float(dictArgs.get("recurrent_dropout").split("\n")[0])

    if 'False' == activationCRF:
        activationCRF = False
    elif activationCRF == 'True':
        activationCRF = True
    if 'False' == convolution_layer:
        convolution_layer = False
    elif convolution_layer == 'True':
        convolution_layer = True

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
    with open(nameFileVocab, encoding="utf-8") as fv:
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


def vectorize(features, tags, vocab, unroll, train, test=False):
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

    if convolution_layer and not test:

        if train:
            add_vocab_ngram(vocab)

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


def min_count_train(vocab, X_train):
    global min_count
    # Count the number of words occurrences.
    occurrences_features = {}
    number_pop = {}

    for feature in range(len(vocab)):
        for key, value in vocab[feature].items():
            if feature not in colIgnore and feature != numColTag:
                if feature not in occurrences_features:
                    occurrences_features[feature] = {}
                    number_pop[feature] = 0
                occurrences_features[feature][value] = 0
                if key == "<unk>" or key == "<pad>":
                    occurrences_features[feature][value] = min_count[feature] ** 2

    for feature in range(len(vocab)):
        if feature not in colIgnore and feature != numColTag:
            for sentence in X_train[feature]:
                for x in sentence:
                    try:
                        occurrences_features[feature][x] += 1
                    except Exception:
                        occurrences_features[feature][x] = 1

    # Replace word which less min_count by <unk>
    for feature in range(len(vocab)):
        if feature not in colIgnore and feature != numColTag:
            for number_sentence in range(len(X_train[feature])):
                for x in range(len(X_train[feature][number_sentence])):
                    if occurrences_features[feature][X_train[feature][number_sentence][x]] < min_count[feature]:
                        key = search_key_dict(vocab[feature], X_train[feature][number_sentence][x])
                        X_train[feature][number_sentence][x] = vocab[feature]["<unk>"]
                        if key in vocab[feature]:
                            number_pop[feature] += 1
    print("Number of word pop with min_count", min_count, number_pop, file=sys.stderr)


def search_key_dict(vocab, value_search):
    for k in vocab:
        if value_search == vocab[k]:
            return k
    return None


def make_model_gru(hidden, embeddings, num_tags, inputs):
    import keras
    from keras.models import Model
    from keras.layers import GRU, Dense, Activation, TimeDistributed, Conv1D, MaxPooling1D, Flatten
    global number_recurrent_layer
    global dropout
    global recurrent_dropout
    global activationCRF

    if convolution_layer:
        conv = TimeDistributed(Conv1D(30, kernel_size=1, padding="same", activation='relu'))(embeddings[-1])
        conv = TimeDistributed(MaxPooling1D(pool_size=2, padding="same"))(conv)
        conv = TimeDistributed(Flatten())(conv)
        # conv = Permute((2, 1))(conv)
        embeddings.pop(len(embeddings) - 1)
        conv = TimeDistributed(Dense(128))(conv)
        embeddings.append(conv)
        x = keras.layers.concatenate(embeddings)
    else:
        x = keras.layers.concatenate(embeddings)
    for recurrent_layer in range(number_recurrent_layer):
        x = GRU(hidden, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)(x)
    if activationCRF:
        from keras_contrib.layers import CRF
        from keras_contrib import losses, metrics
        crf = CRF(num_tags, sparse_target=True)
        x = crf(x)
        model = Model(inputs=inputs, outputs=[x])
        model.compile(loss=losses.crf_loss, optimizer='Nadam', metrics=[metrics.crf_accuracy])
    else:
        x = TimeDistributed(Dense(num_tags))(x)
        x = Activation('softmax')(x)
        model = Model(inputs=inputs, outputs=[x])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['acc'],
                      sample_weight_mode="temporal")  ###############################
    return model


def make_model_bigru(hidden, embeddings, num_tags, inputs, unroll, vocab):
    import keras
    from keras.models import Model
    from keras.layers import GRU, Dense, Activation, TimeDistributed, Bidirectional, Conv1D, MaxPooling1D, Flatten
    global number_recurrent_layer
    global dropout
    global recurrent_dropout
    global activationCRF
    global convolution_layer

    if convolution_layer:
        conv = TimeDistributed(Conv1D(30, kernel_size=1, padding="same", activation='relu'))(embeddings[-1])
        conv = TimeDistributed(MaxPooling1D(pool_size=2, padding="same"))(conv)
        conv = TimeDistributed(Flatten())(conv)
        # conv = Permute((2, 1))(conv)
        embeddings.pop(len(embeddings) - 1)
        conv = TimeDistributed(Dense(128))(conv)
        embeddings.append(conv)
        x = keras.layers.concatenate(embeddings)
    else:
        x = keras.layers.concatenate(embeddings)
    for recurrent_layer in range(number_recurrent_layer):
        x = Bidirectional(GRU(hidden, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout))(x)
    if activationCRF:
        from keras_contrib.layers import CRF
        from keras_contrib import losses, metrics
        crf = CRF(num_tags, sparse_target=True)
        x = crf(x)
        model = Model(inputs=inputs, outputs=[x])
        model.compile(loss=losses.crf_loss, optimizer='Nadam', metrics=[metrics.crf_accuracy])
    else:
        x = TimeDistributed(Dense(num_tags))(x)
        x = Activation('softmax')(x)
        model = Model(inputs=inputs, outputs=[x])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'],
                      sample_weight_mode="temporal")
    model.summary()
    return model


def make_model_lstm(hidden, embeddings, num_tags, inputs):
    import keras
    from keras.models import Model
    from keras.layers import GRU, Dense, Activation, TimeDistributed, Conv1D, MaxPooling1D, Flatten
    global number_recurrent_layer
    global dropout
    global recurrent_dropout
    global activationCRF

    if convolution_layer:
        conv = TimeDistributed(Conv1D(30, kernel_size=1, padding="same", activation='relu'))(embeddings[-1])
        conv = TimeDistributed(MaxPooling1D(pool_size=2, padding="same"))(conv)
        conv = TimeDistributed(Flatten())(conv)
        # conv = Permute((2, 1))(conv)
        embeddings.pop(len(embeddings) - 1)
        conv = TimeDistributed(Dense(128))(conv)
        embeddings.append(conv)
        x = keras.layers.concatenate(embeddings)
    else:
        x = keras.layers.concatenate(embeddings)
    for recurrent_layer in range(number_recurrent_layer):
        x = GRU(hidden, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)(x)
    if activationCRF:
        from keras_contrib.layers import CRF
        from keras_contrib import losses, metrics
        crf = CRF(num_tags, sparse_target=True)
        x = crf(x)
        model = Model(inputs=inputs, outputs=[x])
        model.compile(loss=losses.crf_loss, optimizer='Nadam', metrics=[metrics.crf_accuracy])
    else:
        x = TimeDistributed(Dense(num_tags))(x)
        x = Activation('softmax')(x)
        model = Model(inputs=inputs, outputs=[x])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['acc'],
                      sample_weight_mode="temporal")  ###############################
    return model


def make_model_bilstm(hidden, embeddings, num_tags, inputs):
    import keras
    from keras.models import Model
    from keras.layers import LSTM, Dense, Activation, TimeDistributed, Bidirectional, Conv1D, MaxPooling1D, Flatten
    global number_recurrent_layer
    global dropout
    global recurrent_dropout
    global activationCRF

    if convolution_layer:
        conv = TimeDistributed(Conv1D(30, kernel_size=1, padding="same", activation='relu'))(embeddings[-1])
        conv = TimeDistributed(MaxPooling1D(pool_size=2, padding="same"))(conv)
        conv = TimeDistributed(Flatten())(conv)
        # conv = Permute((2, 1))(conv)
        embeddings.pop(len(embeddings) - 1)
        conv = TimeDistributed(Dense(128))(conv)
        embeddings.append(conv)
        x = keras.layers.concatenate(embeddings)
    else:
        x = keras.layers.concatenate(embeddings)
    for recurrent_layer in range(number_recurrent_layer):
        x = Bidirectional(LSTM(hidden, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout))(x)
    if activationCRF:
        from keras_contrib.layers import CRF
        from keras_contrib import losses, metrics
        crf = CRF(num_tags, sparse_target=True)
        x = crf(x)
        model = Model(inputs=inputs, outputs=[x])
        model.compile(loss=losses.crf_loss, optimizer='Nadam', metrics=[metrics.crf_accuracies])
    else:
        x = TimeDistributed(Dense(num_tags))(x)
        x = Activation('softmax')(x)
        model = Model(inputs=inputs, outputs=[x])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['acc'],
                      sample_weight_mode="temporal")
    return model


def make_modelMWE(hidden, embed, num_tags, unroll, vocab, nbchar=20):
    from keras.layers import Embedding, Input, Reshape
    global recurrent_unit
    global trainable_embeddings
    global nbFeat
    global fasttexts_model
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
            if i - 4 == 2 and i - 4 in embeddingsArgument:
                embedding_matrix, vocab, dimension = loadEmbeddings(vocab, embeddingsArgument[i - 4], i - 4)
                x = Embedding(output_dim=dimension, input_dim=len(vocab[i]), weights=[embedding_matrix],
                              input_length=unroll, trainable=trainable_embeddings)(inputFeat)
            else:
                embedding_matrix, vocab, dimension = loadEmbeddings(vocab, embeddingsArgument[i], i)
                x = Embedding(output_dim=dimension, input_dim=len(vocab[i]), weights=[embedding_matrix],
                              input_length=unroll, trainable=trainable_embeddings)(inputFeat)
        elif i in fasttexts_model:
            if i - 4 == 2 and i - 4 in fasttexts_model:
                embedding_matrix = fasttexts_model[i - 4].matrix_embeddings(vocab[i])
                x = Embedding(output_dim=fasttexts_model[i - 4].size, input_dim=len(vocab[i]), weights=[embedding_matrix],
                              input_length=unroll, trainable=trainable_embeddings)(inputFeat)
            else:
                embedding_matrix = fasttexts_model[i].matrix_embeddings(vocab[i])
                x = Embedding(output_dim=fasttexts_model[i].size, input_dim=len(vocab[i]), weights=[embedding_matrix],
                              input_length=unroll, trainable=trainable_embeddings)(inputFeat)
        elif i in w2v_model :
            if i - 4 == 2 and i - 4 in w2v_model:
                embedding_matrix = w2v_model[i - 4].matrix_embeddings(vocab[i])
                x = Embedding(output_dim=w2v_model[i - 4].size, input_dim=len(vocab[i]), weights=[embedding_matrix],
                              input_length=unroll, trainable=trainable_embeddings)(inputFeat)
            else:
                embedding_matrix = w2v_model[i].matrix_embeddings(vocab[i])
                x = Embedding(output_dim=w2v_model[i].size, input_dim=len(vocab[i]), weights=[embedding_matrix],
                              input_length=unroll, trainable=trainable_embeddings)(inputFeat)
        else:
            if embed.get(i) == 1:
                embedding_matrix = one_hot_vector(vocab[i])
                x = Embedding(output_dim=len(vocab[i]), input_dim=len(vocab[i]), input_length=unroll,
                              trainable=False, weights=[embedding_matrix])(inputFeat)
            else:
                x = Embedding(output_dim=embed.get(i), input_dim=len(vocab[i]), input_length=unroll,
                              trainable=trainable_embeddings)(inputFeat)
        embeddings.append(x)
    if convolution_layer:
        inputFeat = Input(shape=(unroll, nbchar), dtype='int32', name='ngram')
        inputs.append(inputFeat)
        try:
            embedding_matrix = fasttexts_model[1].matrix_embeddings_ngram(vocab[len(vocab) - 1])
            x = Embedding(input_dim=len(vocab[len(vocab) - 1]), output_dim=fasttexts_model[1].size, input_length=(unroll, nbchar),
                          weights=embedding_matrix)(
                inputFeat)
        except Exception:
            x = Embedding(input_dim=len(vocab[len(vocab) - 1]), output_dim=128, input_length=(unroll, nbchar),
                          trainable=trainable_embeddings)(
                inputFeat)
        embeddings.append(x)

    if recurrent_unit == "gru":
        model = make_model_gru(hidden, embeddings, num_tags, inputs)
    elif recurrent_unit == "bigru":
        model = make_model_bigru(hidden, embeddings, num_tags, inputs, unroll, vocab)
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
    nb_unk_vocab = 0
    nb_vector = 0
    with open(filename, encoding="utf8") as fp:
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
                else:
                    nb_unk_vocab += 1
                nb_vector += 1
                embedding[vocab[numColEmbed][word]] = [float(x) for x in tokens[1:]]

                # print("never seen! lenVocab : ",lenVocab," ",len(vocab[numColEmbed]))
    # np.reshape(embedding, (lenVocab, dimension))

    embedding = np.delete(embedding, list(range(lenVocab - 1, len(embedding))), 0)

    return embedding, vocab, dimension


def one_hot_vector(vocab):
    matrix_one_hot = np.zeros((len(vocab), len(vocab)))
    count = 0
    for key, item in vocab.items():
        matrix_one_hot[count][item] = 1
        count += 1
    return matrix_one_hot


def feature_ngram(text, vocab, unroll, nbchar=20):
    X_ngram = []
    print("Add ngrams features", file=sys.stderr)
    for sentence in tqdm(text):
        nb_line = 0
        x_ngram = []
        for line in sentence:
            if line != "\n" and nb_line < unroll:
                word = line.split("\t")[1]
                word_ngram = word_to_ngram(word)
                ngram = np.zeros(nbchar, dtype=np.int32)
                for w in range(len(word_ngram)):
                    if w < nbchar:
                        if word_ngram[w] in vocab:
                            ngram[w] = vocab[word_ngram[w]]
                        else:
                            ngram[w] = vocab["<unk>"]
                nb_line += 1
                x_ngram.append(ngram)
            if line == "\n":
                for i in range(unroll - nb_line):
                    x_ngram.append(np.zeros(nbchar, dtype=np.int32))
        X_ngram.append(np.array(x_ngram))
    return np.array(X_ngram)


def word_to_ngram(word, n_gram=1):
    """
    :param word: to decompose in n_gram
    :param n_gram: to create n_gram of word
    :return: word_n_gram: list of n_gram of word
    """
    word_n_gram = []
    for i in range(len(word)):
        if i + n_gram <= len(word):
            word_n_gram.append(word[i:i + n_gram])
    return word_n_gram


def add_vocab_ngram(vocab, n_gram=1):
    """
    Add vocabulary of n_gram.
    """
    vocab[len(vocab)] = enumdict()
    vocab[len(vocab) - 1]["<unk>"] = 1
    vocab[len(vocab) - 1]["<pad>"] = 0
    for key, word in vocab[1].items():
        for w_n in word_to_ngram(key, n_gram):
            vocab[len(vocab) - 1][w_n]


def n_gram_to_vocab(vocab, word, n_gram=1):
    n_gram_vocab = []

    if word == 0:
        n_gram_vocab.append(vocab[len(vocab) - 1]["<pad>"])
    else:
        for key, item in vocab[1].items():
            if item == word:
                w = key
                continue
        try:
            for w_n in word_to_ngram(w, n_gram):
                n_gram_vocab.append(vocab[len(vocab) - 1][w_n])
        except Exception:
            n_gram_vocab.append(vocab[len(vocab) - 1]["<unk>"])

    return n_gram_vocab


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
    reformatFile.add_deprel_lemma()
    # reformatFile.petits_bateaux_pos_to_ud_pos()

    global colIgnore

    colIgnore = list(range(reformatFile.numberOfColumns))
    for index in args.featureColumns:
        colIgnore.remove(index - 1)
    # colIgnore = uniq(colIgnore)
    colIgnore.sort(reverse=True)

    if validation_data is not None:
        devFile = ReaderCupt(FORMAT, False, True, validation_data, numColTag)
        devFile.run()
        devFile.add_deprel_lemma()

    os.environ['PYTHONHASHSEED'] = '0'
    from numpy.random import seed
    seed(args.numpy_seed)

    import tensorflow as tf
    tf.set_random_seed(args.tensorflow_seed)

    random.seed(args.random_seed)

    from keras import backend as K

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)
    # Force Tensorflow to use a single thread
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

    K.set_session(sess)

    sys.stderr.write("Env session keras : numpy_seed(" + str(args.numpy_seed) + "), tensorflow_seed(" + str(
        args.tensorflow_seed) + "), random_seed(" + str(args.random_seed) + ")..\n")

    if isTrain:

        sys.stderr.write("Load training file..\n")
        features, tags, vocab = load_text_train(reformatFile.resultSequences, vocab)
        min_count_train(vocab, features)
        X, Y, mask, sample_weight = vectorize(features, tags, vocab, unroll, True)
        if convolution_layer:
            X.append(feature_ngram(reformatFile.resultSequences, vocab[len(vocab)-1], unroll))

        # print(X[0].shape, X[-1].shape)
        num_tags = len(vocab[numColTag])

        for i in fasttexts_model.keys():
            if fasttexts_model[i].new_train:
                fasttexts_model[i].train(reformatFile.construct_sentence(i))

        for i in w2v_model.keys():
            if w2v_model[i].new_train:
                w2v_model[i].train(reformatFile.construct_sentence(i))

        sys.stderr.write("Create model..\n")
        model = make_modelMWE(hidden, embed, num_tags, unroll, vocab)
        # from keras.utils import plot_model
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
                    model.fit(X, Y, batch_size=batch, epochs=epochs, shuffle=True)
                else:
                    model.fit(X, Y, batch_size=batch, epochs=epochs, shuffle=True, sample_weight=sample_weight)

                sys.stderr.write("Save model\n")
                model.save(filenameModelWithoutExtension + '.h5')

        else:

            sys.stderr.write("Load dev file..\n")
            if not noreplace:
                for i in fasttexts_model.keys():
                    fasttexts_model[i].similarity_unk_vocab(vocab[i], devFile.resultSequences, i)

            if convolution_layer:
                X_ngram = feature_ngram(devFile.resultSequences, vocab[len(vocab)-1], unroll)
            devFile.verifyUnknowWord(vocab)
            features, tags, useless = load_text_test(devFile.resultSequences, vocab)

            X_test, Y_test, mask, useless = vectorize(features, tags, vocab, unroll, False, test=True)
            if convolution_layer:
                X_test.append(X_ngram)

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
        if not noreplace:
            for i in fasttexts_model.keys():
                fasttexts_model[i].similarity_unk_vocab(vocab[i], reformatFile.resultSequences, i)

        #reformatFile.verifyUnknowWord(vocab)
        sys.stderr.write("Load model..\n")
        from keras.models import load_model

        if activationCRF:
            from keras_contrib.utils import save_load_utils
            if convolution_layer:
                vocab.pop(len(vocab) - 1)
            num_tags = len(vocab[args.mweTags - 1])
            model = make_modelMWE(hidden, embed, num_tags, unroll, vocab)
            save_load_utils.load_all_weights(model, filenameModelWithoutExtension + '.h5', include_optimizer=False)
            #from keras_contrib.layers import CRF
            #from keras.models import load_model
            #from keras_contrib.losses import crf_loss
            #from keras_contrib.metrics import crf_accuracy
            #custom_objects = {'CRF': CRF,
            #                  'crf_loss': crf_loss,
            #                  'crf_accuracy': crf_accuracy}
            #model = load_model(filenameModelWithoutExtension + '.h5',
            #                          custom_objects=custom_objects)
        else:
            model = load_model(filenameModelWithoutExtension + '.h5')

        # model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['acc'],
        #              sample_weight_mode="temporal")
        sys.stderr.write("Load testing file..\n")
        # reformatFile.petits_bateaux_pos_to_ud_pos()
        if convolution_layer:
            X_ngram_test = feature_ngram(reformatFile.resultSequences, vocab[len(vocab) - 1], unroll)
        reformatFile.verifyUnknowWord(vocab)
        features, tags, useless = load_text_test(reformatFile.resultSequences, vocab)
        X, Y, mask, useless = vectorize(features, tags, vocab, unroll, False, test=True)
        if convolution_layer:
            X.append(X_ngram_test)
        classes = model.predict(X)
        # sys.stderr.write(classes.shape+ "\nclasses: "+ classes)

        prediction = maxClasses(classes, Y, unroll, mask)

        sys.stderr.write("Add prediction...\n")
        pred, listNbToken = genereTag(prediction, vocab, unroll)
        reformatFile.addPrediction(pred, listNbToken)
        if args.file_save_cupt:
            reformatFile.saveFileCupt(args.file_save_cupt)
        else:
            reformatFile.printFileCupt()

        # print(len(pred))
        sys.stderr.write("END testing\t")
        print(str(datetime.datetime.now()), file=sys.stderr)

    else:
        sys.stderr("Error argument: Do you want to test or train ?")
        exit(-2)


main()
