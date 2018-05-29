#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys

r"""
    Reader class to read cupt file.
    Give a boolean test and a FORMAT in [BIO, BIOg, BIOcat, BIOgcat, IO, IOg, IOcat, IOgcat] to initialize it.
"""


def error_format():
    sys.stderr("Error with FORMAT option : add manually the new format.")
    exit(-1)


class ReaderCupt:

    def __init__(self, FORMAT, test, file):
        self.file = file
        self.resultSequences = []
        self.TagBegin = "0"
        self.TagInside = "0"
        self.TagOuside = "0"
        self.TagGap = "0"
        self.withVMWE = False
        self.FORMAT = FORMAT
        self.test = test
        self.initialize_parameters()

    r"""
        Run of this class to read file and extract columns which are important for RNN.
    """

    def run(self):
        self.read()

    r"""
        Initialize tag to the good format in [BIO, BIOg, BIOcat, BIOgcat, IO, IOg, IOcat, IOgcat]
        to read a .cupt and transform it.
    """

    def initialize_parameters(self):
        # BIO format without gap
        if self.FORMAT == "BIO":
            self.TagBegin = "B"
            self.TagInside = "I"
            self.TagOuside = "O"
            self.TagGap = "O"
        # IO format without gap
        elif self.FORMAT == "IO":
            self.TagBegin = "I"
            self.TagInside = "I"
            self.TagOuside = "O"
            self.TagGap = "O"
        # BIO format with gap
        elif self.FORMAT == "BIOg":
            self.TagBegin = "B"
            self.TagInside = "I"
            self.TagOuside = "O"
            self.TagGap = "g"
        # IO format with gap
        elif self.FORMAT == "IOg":
            self.TagBegin = "I"
            self.TagInside = "I"
            self.TagOuside = "O"
            self.TagGap = "g"
        # BIO without gap but with VMWE-tagger
        elif self.FORMAT == "BIOcat":
            self.TagBegin = "B"
            self.TagInside = "I"
            self.TagOuside = "O"
            self.TagGap = "O"
            self.withVMWE = True
        # IO without gap but with VMWE-tagger
        elif self.FORMAT == "IOcat":
            self.TagBegin = "I"
            self.TagInside = "I"
            self.TagOuside = "O"
            self.TagGap = "O"
            self.withVMWE = True
        # BIO with gap and VMWE-tagger
        elif self.FORMAT == "BIOgcat":
            self.TagBegin = "B"
            self.TagInside = "I"
            self.TagOuside = "O"
            self.TagGap = "g"
            self.withVMWE = True
        # IO with gap and VMWE-tagger
        elif self.FORMAT == "IOgcat":
            self.TagBegin = "I"
            self.TagInside = "I"
            self.TagOuside = "O"
            self.TagGap = "g"
            self.withVMWE = True
        # Error in the format
        else:
            error_format()

    r"""
        Return if a file is completely read.
    """

    def fileCompletelyRead(self, line):
        return line == ""

    r"""
        Return if is not the and of the sentence.
    """

    def isInASequence(self, line):
        return line != "\n" and line != ""

    r"""
        Return if the line is a comment.
    """

    def lineIsAComment(self, line):
        return line[0] == "#"

    r"""
        Read and transform in the good format in [BIO, BIOg, BIOcat, BIOgcat, IO, IOg, IOcat, IOgcat].
    """

    def read(self):
        line = self.file.readline()
        while not self.fileCompletelyRead(line):
            sequenceCupt = []
            while self.isInASequence(line):
                while self.lineIsAComment(line):
                    line = self.file.readline()
                sequenceCupt.append(line.rstrip().split("\t"))
                line = self.file.readline()
            self.createSequence(sequenceCupt, self.test)
            line = self.file.readline()

    r"""
        create a Sequence with in the new format.
    """

    def createSequence(self, sequenceCupt, test):
        startVMWE = False
        comptUselessID = 1
        sequences = []
        if not test:
            numberVMWE = self.numberVMWEinSequence(sequenceCupt)
        else:
            numberVMWE = 1

        for index in range(numberVMWE):
            listVMWE = {}  # self.createListSequence(sequenceCupt)
            for sequence in sequenceCupt:
                tagToken = ""
                tag = sequence[-1].split(";")[index % len(sequence[-1].split(";"))]
                if sequence[-1] != "*" and not "-" in sequence[0] and not "." in sequence[0]:
                    # update possible for many VMWE on one token
                    if len(tag.split(":")) > 1:
                        indexVMWE = tag.split(":")[0]
                        VMWE = tag.split(":")[1]
                        listVMWE[indexVMWE] = sequence[0] + ":" + VMWE
                        if self.withVMWE:
                            tagToken += self.TagBegin + VMWE + "\t0"
                        else:
                            tagToken += self.TagBegin + "" + "\t0"
                    elif listVMWE.has_key(tag):
                        indexVMWE = listVMWE.get(tag).split(":")[0]
                        if self.withVMWE:
                            VMWE = listVMWE.get(tag).split(":")[1]
                        else:
                            VMWE = ""
                        tagToken += self.TagInside + VMWE + "\t" + indexVMWE
                    elif self.endVMWE(int(sequence[0]) + comptUselessID, sequenceCupt, listVMWE):
                        tagToken += self.TagGap + "\t0"
                    else:
                        tagToken += self.TagOuside + "\t0"

                elif startVMWE and sequence[-1] == "*":
                    tagToken += self.TagGap + "\t0"
                elif not startVMWE and sequence[-1] == "*" or sequence[-1] == "_":
                    tagToken += self.TagOuside + "\t0"

                if "-" in sequence[0] or "." in sequence[0]:
                    comptUselessID += 1
                if not "-" in sequence[0] and not "." in sequence[0]:
                    startVMWE = self.endVMWE(int(sequence[0]) + comptUselessID, sequenceCupt, listVMWE)

                newSequence = sequence[0] + "\t" + sequence[1] + "\t"
                # Lemma == _
                if sequence[2] == "_":
                    newSequence += sequence[1] + "\t"
                else:
                    newSequence += sequence[2] + "\t"
                # UPOS == _
                if sequence[3] == "_":
                    newSequence += sequence[4] + "\t"
                else:
                    newSequence += sequence[3] + "\t"

                sequences.append(newSequence + tagToken + "\t\t\t_")
            sequences.append("\n")
            self.resultSequences.append(sequences)

    r"""
        Verify if the end of overlaps 
    """

    def endVMWE(self, param, sequenceCupt, listVWME):
        for index in range(param, len(sequenceCupt)):
            tag = sequenceCupt[index][-1].split(";")[0]

            if tag == "*":
                continue
            if listVWME.has_key(tag.split(":")[0]):
                return True
        return False

    r"""
        Count VMWE numbers in the sequence  which is on one token
    """

    def numberVMWEinSequence(self, sequenceCupt):
        numberVMWE = 1
        for sequence in sequenceCupt:
            if sequence[-1] == "*" or sequence[-1] == "_":
                continue

            if len(sequence[-1].split(";")) > numberVMWE:
                numberVMWE = len(sequence[-1].split(";"))
        return numberVMWE

    """def verifyVocab(self, vocabTrain):
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
        print(newTestFile)"""

    def saveVocab(self, nameFileVocab, vocab):
        file = open(nameFileVocab, "w")
        for key, voc in vocab.items():
            for keyToken, valVoc in voc.items():
                file.write(str(keyToken) + ":" + str(valVoc) + "\n")

    def verifyUnknowWord(self, vocab):
        for sentence in self.resultSequences:
            for line in range(len(sentence)):
                if not self.isInASequence(sentence[line]):
                    pass
                else:
                    lineTMP = sentence[line].split("\t")
                    print(lineTMP)
                    for index in range(len(lineTMP)):
                        if not vocab.has_key(lineTMP[index]):
                            lineTMP[index] = "<unk>"
                    newLine = lineTMP[0] + "\t" + lineTMP[1] + "\t" + lineTMP[2] + "\t" + lineTMP[3] + "\t" + lineTMP[4] + "\t" + lineTMP[5] + "\t\t\t_"
                    sentence[line] = newLine

    def loadVocab(self, nameFileVocab):
        vocab = dict()
        with open(nameFileVocab) as fv:
            for line in fv:
                if self.fileCompletelyRead(line):
                    pass
                elif not self.isInASequence(line):
                    pass
                else:
                    vocab[line.split(":")[0]] = int(line.split(":")[1])

        return vocab