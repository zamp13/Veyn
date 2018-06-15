#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import sys

from reader import ReaderCupt

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

class Main():

    def __init__(self, args):
        self.args = args

    def run(self):
        print ReaderCupt("IO", self.args.test, self.args.fileCupt).run()


if __name__ == "__main__":
    Main(parser.parse_args()).run()
