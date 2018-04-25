#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import sys

parser = argparse.ArgumentParser(description="""
        ENTRE : fichier au format BIO -->
        SORTIE: ecriture au format .cupt stdout

        FORMAT .cupt:
        ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE\n

        FORMAT .dimsum:
        ID FORM LEMMA UPOS FEATS BIO refBIO _
        """)
parser.add_argument("--cupt", metavar="fileCupt", dest="fileCupt",
                    required=True, type=argparse.FileType('r'),
                    help="""The cupt-standard file""")
parser.add_argument("--dimsum", metavar="fileDimsum", dest="fileDimsum",
                    required=True, type=argparse.FileType('r'),
                    help="""The BIO-standard file""")


class Main():

    def __init__(self, args):
        self.args = args

    def run(self):

        fileDimsum = self.args.fileDimsum
        fileCupt = self.args.fileCupt

        lineD = fileDimsum.readline()
        lineC = fileCupt.readline()

        listTag = {}
        cpt = 0
        isVMWE = False
        while (not (self.fileCompletelyRead(lineD)) or not (self.fileCompletelyRead(lineC))):
            # align text
            while (self.lineIsAComment(lineC)):
                print(lineC, end='')
                lineC = fileCupt.readline()

            while (self.lineIsAComment(lineD)):
                lineD = fileDimsum.readline()

            # if end of sequence
            if (lineD == "\n" and lineC == "\n"):
                print()
                lineD = fileDimsum.readline()
                lineC = fileCupt.readline()
                cpt = 0
                isVMWE = False
                listTag = {}
                continue

            # find tag in dimsum
            lineD = lineD.split("\t")
            tag, cpt, isVMWE = self.findTag(lineD, cpt, listTag, isVMWE)

            # print the tag at the good format
            lineC = lineC.split("\t")
            newLine = ""
            for index in range(len(lineC) - 1):
                newLine += lineC[index] + "\t"
            newLine += tag
            print(newLine)
            lineD = fileDimsum.readline()
            lineC = fileCupt.readline()

    def fileCompletelyRead(self, lineD):
        return lineD == ""

    def isInASequence(self, lineD):
        return lineD != "\n" and lineD != ""

    def lineIsAComment(self, lineD):
        return lineD[0] == "#"

    def findTag(self, lineD, cpt, listTag, isVMWE):
        tag = lineD[4]

        if tag == "O" or "-" in lineD[0] or "." in lineD[0]:
            tag = "*"
            isVMWE = False
            return tag, cpt, isVMWE

        if tag == "o":
            tag = "*"
            isVMWE = True
            return tag, cpt, isVMWE

        if tag[0] == "B":
            isVMWE = True
            cpt += 1
            listTag[tag] = str(cpt)
            tag = str(cpt) + ":" + tag
            return tag, cpt, isVMWE

        if tag[0] == "I":
            tag = tag[1:-1] + tag[-1]
            if listTag.has_key(tag) and isVMWE:
                tag = listTag.get(tag)
            else:
                tag = "*"
            return tag, cpt, isVMWE

        sys.stderr.write("Error with tags predict : {0} \n".format(tag))
        exit(1)


if __name__ == "__main__":
    Main(parser.parse_args()).run()
