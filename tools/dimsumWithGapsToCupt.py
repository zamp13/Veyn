#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
import fileinput
import sys
import re


def fileCompletelyRead(lineD):
    return lineD == ""


def isInASequence(lineD):
    return lineD != "\n" and lineD != ""


def lineIsAComment(lineD):
    return lineD[0] == "#"


def findTag(lineD, cpt, numPreviousB, previousTag):
    tag = lineD[4]
    if (tag.upper() == "O"):
        return "*", cpt, numPreviousB
    if (tag[0] == "B"):
        cpt += 1
        tag = str(cpt) + ":CRF"
        numPreviousB = cpt
        return tag, cpt, numPreviousB
    if (tag[0] == "I"):
        if (numPreviousB == 0):
            cpt += 1
            tag = str(cpt) + ":CRF"
            return tag, cpt, numPreviousB
        else:
            return str(numPreviousB), cpt, numPreviousB
    if (tag[0] == "b"):
        cpt += 1
        tag = str(cpt) + ":CRF"
        return tag, cpt, numPreviousB
    if (tag[0] == "i"):
        return str(cpt), cpt, numPreviousB

    sys.stderr.write("Error with tags predict : {0} \n".format(tag))
    exit(1)


def main():
    if (len(sys.argv) != 3):
        print("Error, give a dimsum file and a cupt file.")
        exit(1)

    filepathDimsum = sys.argv[1]
    filepathCupt = sys.argv[2]

    fileDimsum = open(filepathDimsum, "r")
    fileCupt = open(filepathCupt, "r")

    lineD = fileDimsum.readline()
    lineC = fileCupt.readline()

    previousTag = "*"
    numPreviousB = 0
    cpt = 0

    while (not (fileCompletelyRead(lineD)) or not (fileCompletelyRead(lineC))):
        # align text
        while (lineIsAComment(lineC)):
            #print(lineC)
            lineC = fileCupt.readline()

        while (lineIsAComment(lineD)):
            lineD = fileDimsum.readline()

        # if end of sequence
        if (lineD == "\n" and lineC == "\n"):
            print()
            lineD = fileDimsum.readline()
            lineC = fileCupt.readline()
            cpt = 0
            previousTag = "*"
            continue


        # find tag in dimsum
        lineD = lineD.split("\t")
        tag, cpt, numPreviousB = findTag(lineD, cpt, numPreviousB, previousTag)

        # print the tag at the good format
        lineC = lineC.split("\t")
        newLine = ""
        for index in range(len(lineC)-1):
            newLine += lineC[index] + "\t"
        newLine += tag
        print(newLine)
        previousTag = tag
        lineD = fileDimsum.readline()
        lineC = fileCupt.readline()


main()
