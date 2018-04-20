#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import sys

parser = argparse.ArgumentParser(description=""" Add prediction into a dimsum file (default=stdout).
        Give two arguments, dimsum file and file who contain just tag of VMWE at the format BIO.""")
parser.add_argument("--dimsum", metavar="fileCupt", dest="fileCupt",
                    required=True, type=argparse.FileType('r'),
                    help="""The cupt-standard file""")
parser.add_argument("--tag", metavar="fileTags", dest="fileTags",
                    required=True, type=argparse.FileType('r'),
                    help="""The tag-standard file""")
class Main():
    def __init__(self, args):
        self.args= args

    def run(self):
        dimsum = self.args.fileDimsum
        tags = self.args.fileTags

        lineD = dimsum.readline()
        lineT = tags.readline()
        while lineT != "PREDICT":
            lineT = tags.readline()
        lineT = tags.readline()
        FlaglineTEmpty = False

        while (not (self.fileCompletelyRead(lineD) or self.fileCompletelyRead(lineT))):
            if (self.lineIsAComment(lineD)):
                print(lineD[:-1])
                lineD = dimsum.readline()
                continue
            if (self.lineIsAComment(lineT)):
                lineT = tags.readline()
                continue

            if (lineD == "\n" or lineT == "\n"):
                if (lineD == "\n" and lineT == "\n"):
                    print()
                    lineD = dimsum.readline()
                    lineT = tags.readline()
                    continue
                elif (lineT == "\n" and not (lineD == "\n")):
                    FlaglineTEmpty = True
                else:
                    print("------->", lineD, "-------->", lineT, file=sys.stderr)
                    sys.stderr.write("Error, two files not align?\n")
                    exit(1)

            lineD = lineD[:-1].split("\t")
            newLine = ""
            continueNext = False

            for i in range(len(lineD)):
                if (continueNext):
                    continueNext = False
                    continue
                if (i == 4):
                    if (FlaglineTEmpty):
                        newLine += "O\t0\t"
                    else:
                        newLine += lineT[:-1] + "\t0\t"
                    continueNext = True
                else:
                    newLine += lineD[i] + "\t"

            print(newLine)

            lineD = dimsum.readline()
            if (not FlaglineTEmpty):
                lineT = tags.readline()
            FlaglineTEmpty = False

    def fileCompletelyRead(self,line):
        return line == ""


    def isInASequence(self,line):
        return line != "\n" and line != ""


    def lineIsAComment(self,line):
        return line[0] == "#"


