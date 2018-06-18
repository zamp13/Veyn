#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import collections
import sys

from reader import ReaderCupt, fileCompletelyRead, isInASequence

parser = argparse.ArgumentParser(description="""
        ENTRE : fichier au format.cupt -->
        SORTIE: ecriture au format BIO stdout

        FORMAT .cupt:
        ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE\n

        FORMAT .dimsum:
        ID FORM LEMMA UPOS FEATS IVMWE refBIO _
        """)
parser.add_argument("--cupt", metavar="fileCupt", dest="fileCupt",
                    required=True, type=argparse.FileType('r'),
                    help="""The cupt-standard file""")
parser.add_argument("--test", action='store_const', const=True,
                    dest='test',
                    help="""If the file is a fileTest""")
parser.add_argument('--list', nargs='+', type=int, dest="list")

r"""
    Load and return the vocabulary dict.
"""
def enumdict():
    a = collections.defaultdict(lambda: len(a))
    return a

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

class Main:

    def __init__(self, args):
        self.args = args

    def run(self):
        reader = ReaderCupt("BIOgcat", True, self.args.test, self.args.fileCupt, 10)
        reader.run()
        vocab = loadVocab("Model/train-BIO-refac-cupt.voc")
        reader.verifyUnknowWord(vocab)
        reader.printResultSequence()


if __name__ == "__main__":
    Main(parser.parse_args()).run()
