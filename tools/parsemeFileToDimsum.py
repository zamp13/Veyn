#!/usr/bin/python
# -*- coding: utf-8 -*-
'''PSEUDO CODE :

ENTRE : fichier au format.cupt
SORTIE: ecriture au format BIO ( .dimsum )

FORMAT .cupt:
ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE

FORMAT .dimsum
ID FORM LEMMA UPOS FEATS BIO refBIO _

'''
import sys


def fileCompletelyRead(line):
    return line == ""


def isInASequence(line):
    return line != "\n" and line != ""


def lineIsAComment(line):
    return line[0] == "#"

def initListVMWES(sequence):
    """
    Initialize a list of VMWES of sequence
    :param sequence:
    :return: list of VMWES and list of index who having MWE
    """
    listVMWES = dict()
    listOfIndexHavingMWE = []
    for i in range(len(sequence)):
        token = sequence[i]
        VMWES = token[-1]

        if (VMWES == "*"):
            continue

        indexToken = token[0]
        listOfIndexHavingMWE.append(indexToken)

        VMWES = VMWES.split(";")
        for VMWE in VMWES:
            if (":" in VMWE):
                VMWE = VMWE.split(":")[0]
            if (listVMWES.has_key(VMWE)):
                listVMWES[VMWE] += "\t" + indexToken
            else:
                listVMWES[VMWE] = indexToken

    return listVMWES, listOfIndexHavingMWE


def separateVMWEsOverlapsOrNot(listVMWES):
    listVMWEsOverlaps = dict()
    listVMWEsNotOverlaps = dict()
    listIndexOverlaps = []
    listIndexNotOverlaps = []
    overlap = False

    for key, indexes in listVMWES.iteritems():
        indexes = indexes.split("\t")
        for index in indexes:
            if (index in listIndexNotOverlaps):
                listIndexOverlaps.append(index)
                listIndexNotOverlaps.remove(index)
            elif (not (index in listIndexNotOverlaps)):
                listIndexNotOverlaps.append(index)

    for key, indexes in listVMWES.iteritems():
        splitIndexes = indexes.split("\t")
        for index in splitIndexes:
            if (index in listIndexOverlaps):
                listVMWEsOverlaps[key] = indexes
                overlap = True
                break
        if (not overlap):
            listVMWEsNotOverlaps[key] = indexes
        overlap = False

    return listVMWEsOverlaps, listVMWEsNotOverlaps

def verifyOverlaps(listVMWES):
    listIndex = []

    for key, indexes in listVMWES.iteritems():
        indexes = indexes.split("\t")
        for index in indexes:
            if(index in listIndex):
                return True
            else:
                listIndex.append(index)
    return False


def handleOverlaps(sequence, listVMWES):
    listVMWEsOverlaps, listVMWEsNotOverlaps = separateVMWEsOverlapsOrNot(listVMWES)
    # print("---------------------------------\n",listVMWES,"\n\n\n",listVMWEsOverlaps,"-----------------------------------\n")
    for key, indexes in listVMWEsOverlaps.iteritems():
        listVMWEsNotOverlaps[key] = indexes
        handleSimpleSequence(sequence,listVMWEsNotOverlaps)
        listVMWEsNotOverlaps.pop(key)
        break

def isPartOfaVMWE(currentIndex, listVMWES):
    for VMWE, indexes in listVMWES.iteritems():
        indexes = indexes.split("\t")
        for index in indexes:
            index = int(index)
            if(currentIndex == index):
                return True
    return False

def isVMWEbeginning(currentIndex, listVMWES):
    for VMWE, indexes in listVMWES.iteritems():
        firstIndex = indexes.split("\t")[0]
        firstIndex = int(firstIndex)
        if(currentIndex == firstIndex):
            return True
    return False

def findPrevIndex(currentIndex, listVMWES):
    for VMWE, indexes in listVMWES.iteritems():
        prevIndex = "0"
        indexes = indexes.split("\t")
        for index in indexes:
            index = int(index)
            if(currentIndex == index):
                return prevIndex
            prevIndex = index
    return None

def isVMWEending(currentIndex, listVMWES):
    for VMWE, indexes in listVMWES.iteritems():
        indexes = indexes.split("\t")
        lastIndex = indexes[len(indexes)-1]
        lastIndex = int(lastIndex)
        if(currentIndex == lastIndex):
            return True
    return False


def defineTagForEachIndex(listVMWES, nbTokens):
    tagsOfIndex = dict()
    deep = 0  # define in how many other VMWES is the current token

    for index in range(1, nbTokens + 1):  # we look for each index of the sequence
        if (isPartOfaVMWE(index, listVMWES)):
            if (isVMWEbeginning(index, listVMWES)):
                deep += 1
                tagsOfIndex[index] = "B" + str(deep) + "\t0"
            else:
                prevIndex = findPrevIndex(index, listVMWES)
                tagsOfIndex[index] = "I" + str(deep) + "\t" + str(prevIndex)
            if (isVMWEending(index, listVMWES)):
                deep -= 1
        elif (deep > 0):
            tagsOfIndex[index] = "o" + "\t0"
        else:
            tagsOfIndex[index] = "O" + "\t0"

    return tagsOfIndex


def handleSimpleSequence(sequence, listVMWES):
    nbTokens = int(sequence[len(sequence) - 1][0])
    tagsOfIndex = defineTagForEachIndex(listVMWES, nbTokens)

    sequencePrint = ""

    for i in range(len(sequence)):
        tokenP = sequence[i]
        indexCurrentToken = tokenP[0]

        if "-" in indexCurrentToken or "." in indexCurrentToken:
            tag = "O" + "\t0"

        elif (tagsOfIndex.has_key(int(indexCurrentToken))):
            tag = tagsOfIndex.get(int(indexCurrentToken))
        else:
            sys.stderr.write("Error\n")
            print(tagsOfIndex, indexCurrentToken)
            exit(1)

        tokenLine = tokenP[0] + "\t" + tokenP[1] + "\t" + tokenP[2] + "\t" + tokenP[3] + "\t" + tag + "\t\t\t_\n"
        sequencePrint = sequencePrint + tokenLine
    print(sequencePrint)  # add an empty line between each sequence


def handleSequence(sequence):
    listVMWES, listOfIndexHavingMWE = initListVMWES(sequence)

    if (verifyOverlaps(listVMWES)):
        handleOverlaps(sequence, listVMWES)
    else:
        handleSimpleSequence(sequence, listVMWES)

def main():
    """
    Main of this file
    :arg file's name in format .cupt
    """
    if (len(sys.argv) != 2):
        print("Error, give a file with .cupt format.")
        exit(1)

    fileCupt = sys.argv[1]

    cupt = open(fileCupt, "r")

    line = cupt.readline()
    while ( not fileCompletelyRead(line) ):
        sequence = []
        while (isInASequence(line)):
            while (lineIsAComment(line)):
                line = cupt.readline()
            sequence.append(line.rstrip().split("\t"))
            line = cupt.readline()
        handleSequence(sequence)
        # return                 ########################################## <----
        line = cupt.readline()

main()
