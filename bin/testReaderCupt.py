#! /usr/bin/env python3
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
