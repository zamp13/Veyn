#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser(description="""Verify and vectorise if all token of testFile is in vocab trainFile""")

parser.add_argument("--train", metavar="fileTrain", dest="fileTrain",
                    required=True, type=argparse.FileType('r'),
                    help="""The train BIO file""")
parser.add_argument("--test", metavar="fileTest", dest="fileTest",
                    required=True, type=argparse.FileType('r'),
                    help="""The test BIO file""")


class Main():

    def __init__(self, args):
        self.args = args

    def run(self):

        fileTrain = self.args.fileTrain
        fileTest = self.args.fileTest

        vocabTrain = self.load_vocab_train(fileTrain)
        self.generate_new_sequence_test(fileTest, vocabTrain)

    def load_vocab_train(self, fileTrain):
        vocab = {}
        with fileTrain as fp:
            for line in fp:
                if line == "\n" or line[0] == "#":
                    continue
                else:
                    sequence = line.strip().split("\t")
                    vocab[sequence[1]] = 0
                    vocab[sequence[2]] = 0
        return vocab

    def generate_new_sequence_test(self, fileTest, vocabTrain):
        newTestFile = ""
        with fileTest as fp:
            for line in fp:
                if line == "\n":
                    newTestFile += line
                elif "#" in line:
                    newTestFile += line
                elif line.split("\t")[1] in vocabTrain or line.split("\t")[2] in vocabTrain:
                    newTestFile += line
                else:
                    sequence = line.split("\t")
                    newTestFile += sequence[0] + "\t<unk>\t<unk>\t_\t_\t\t\t_\n"
        print(newTestFile)

if __name__ == "__main__":
    Main(parser.parse_args()).run()
