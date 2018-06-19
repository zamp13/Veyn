#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import collections
import datetime
import sys

import numpy as np

from reader import ReaderCupt, fileCompletelyRead, isInASequence

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
                    Careful! You can't have a column in common with ignoreColumns.
                    """)
parser.add_argument("--file", metavar="filename", dest="filename", required=True, type=argparse.FileType('r'),
                    help="""
                    Give a file in the Extended CoNLL-U (.cupt) format.
                    You can only give one file to train/test a model.
                    """)
parser.add_argument("--mode", type=str, dest='mode', required=True,
                    help="""
                    To choice the mode of the system : train/test.
                    If the file is a train file and you want to create a model use \'train\'.
                    If the file is a test/dev file and you want to load a model use \'test\'.
                    """)
parser.add_argument("--model", action='store', type=str,
                    required=True, dest='model',
                    help="""
                    Name of the model which you want to save/load without extension.
                    e.g \'nameModel\' , and the system save/load files nameModel.h5, nameModel.json and nameModel.voc.
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

    numColTag = args.mweTags - 1
    colIgnore = range(11)
    filename = args.filename
    filenameModelWithoutExtension = args.model

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

    if args.mode.lower() == "train":
        isTrain = True
        isTest = False
    elif args.mode.lower() == "test":
        isTrain = False
        isTest = True
    else:
        sys.stderr.write("Error with argument --mode (train/test)")
        exit(-10)

    if args.io:
        FORMAT = "IO"
    else:
        FORMAT = "BIO"

    if not args.nogap:
        FORMAT += "g"

    if args.withMWE:
        FORMAT += "cat"

    for index in args.featureColumns:
        colIgnore.remove(index - 1)
    colIgnore = uniq(colIgnore)
    colIgnore.sort(reverse=True)


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
    import keras
    from keras.models import Model, load_model
    from keras.layers import Embedding, Input, GRU, Dense, Activation, TimeDistributed, \
        Bidirectional

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
                prediction[i][j][0] = maxTag

    return prediction


def genereTag(prediction, vocab, unroll):
    rev_vocabTags = {i: char for char, i in vocab[numColTag].items()}
    #sys.stderr.write(str(rev_vocabTags) + "\n")
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
        #sys.stderr.write("\n")
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
    unroll = args.max_sentence_size
    validation_split = args.validation_split
    validation_data = args.validation_data
    embed = 64
    epochs = args.epoch
    vocab = []

    sys.stderr.write("Load FORMAT ..\t")
    print(str(datetime.datetime.now()), file=sys.stderr)
    reformatFile = ReaderCupt(FORMAT, args.withOverlaps, isTest, filename, numColTag)
    reformatFile.run()

    if validation_data is not None:
        devFile = ReaderCupt(FORMAT, False, True, validation_data, numColTag)
        devFile.run()

    if isTrain:

        sys.stderr.write("Load training file..\n")
        features, tags, vocab = load_text_train(reformatFile.resultSequences, vocab)
        X, Y, mask, sample_weight = vectorize(features, tags, vocab, unroll)

        num_tags = len(vocab[numColTag])

        sys.stderr.write("Create model..\n")
        model = make_modelMWE(hidden, embed, num_tags, unroll, vocab)
        # plot_model(model, to_file='modelMWE.png', show_shapes=True)

        if validation_data is None:
            sys.stderr.write("Starting training with validation_split...\n")
            model.fit(X, Y, batch_size=batch, epochs=epochs, shuffle=True,
                      sample_weight=sample_weight, validation_split=validation_split)
        else:
            sys.stderr.write("Load dev file..\n")
            devFile.verifyUnknowWord(vocab)
            features, tags, useless = load_text_test(devFile.resultSequences, vocab)

            X_test, Y_test, mask, useless = vectorize(features, tags, vocab, unroll)

            sys.stderr.write("Starting training with validation_data ...\n")
            model.fit(X, Y, batch_size=batch, epochs=epochs, shuffle=True,
                      validation_data=(X_test, Y_test), sample_weight=sample_weight)

        sys.stderr.write("Save vocabulary...\n")

        reformatFile.saveVocab(filenameModelWithoutExtension + '.voc', vocab)

        sys.stderr.write("Save model..\n")
        # model to HDF5
        model.save(filenameModelWithoutExtension + '.h5')

        sys.stderr.write("END training\t")
        print(str(datetime.datetime.now()) + "\n", file=sys.stderr)

    elif isTest:

        sys.stderr.write("Load vocabulary...\n")
        vocab = loadVocab(filenameModelWithoutExtension + ".voc")
        reformatFile.verifyUnknowWord(vocab)

        sys.stderr.write("Load model..\n")
        from keras.models import load_model

        # Use statefull GRU with SGRU
        # Load weights into the new model
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
