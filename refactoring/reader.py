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

    def __init__(self, FORMAT, withOverlaps, test, file):
        self.file = file
        self.fileCupt = []
        self.resultSequences = []
        self.TagBegin = "0"
        self.TagInside = "0"
        self.TagOuside = "0"
        self.TagGap = "0"
        self.withVMWE = False
        self.withOverlaps = withOverlaps
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
                    self.fileCupt.append(line)
                    line = self.file.readline()
                sequenceCupt.append(line.rstrip().split("\t"))
                self.fileCupt.append(line)
                line = self.file.readline()
            if self.withOverlaps and not self.test:
                self.createSequenceWithOverlaps(sequenceCupt)
            else:
                self.createSequence(sequenceCupt)

            self.fileCupt.append(line)
            line = self.file.readline()

    r"""
        create a Sequence without overlaps in the new format.
    """

    def createSequence(self, sequenceCupt):
        startVMWE = False
        comptUselessID = 1
        sequences = []

        listVMWE = {}  # self.createListSequence(sequenceCupt)
        for sequence in sequenceCupt:
            tagToken = ""
            tag = sequence[-1].split(";")[0]
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
        if sequences not in self.resultSequences:
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

    r"""
        Save vocab dict into nameFileVocab
    """

    def saveVocab(self, nameFileVocab, vocab):
        file = open(nameFileVocab, "w")
        for key, voc in vocab.items():
            for keyToken, valVoc in voc.items():
                file.write(str(keyToken) + "<|>" + str(valVoc) + "\n")
            file.write("\n")

    r"""
        Verify if words in resultSequences is in the vocab. 
    """

    def verifyUnknowWord(self, vocab):
        for sentence in self.resultSequences:

            for line in range(len(sentence)):
                if not self.isInASequence(sentence[line]):
                    pass
                else:
                    lineTMP = sentence[line].split("\t")
                    if not vocab.has_key(lineTMP[1]):
                        lineTMP[1] = "<unk>"
                    if not vocab[2].has_key(lineTMP[2]):
                        lineTMP[2] = "<unk>"
                    if not vocab[3].has_key(lineTMP[3]):
                        lineTMP[3] = "<unk>"
                    newLine = lineTMP[0] + "\t" + lineTMP[1] + "\t" + lineTMP[2] + "\t" + lineTMP[3] + "\t" + lineTMP[
                        4] + "\t" + lineTMP[5] + "\t\t\t_"
                    sentence[line] = newLine

    r"""
        Load and return the vocabulary dict.
    """
    def loadVocab(self, nameFileVocab):
        vocab = dict()
        index = 0
        vocab[index] = dict()
        with open(nameFileVocab) as fv:
            for line in fv:
                if self.fileCompletelyRead(line):
                    pass
                elif not self.isInASequence(line):
                    index += 1
                    vocab[index] = dict()
                else:
                    vocab[index][line.split("<|>")[0]] = int(line.split("<|>")[1])

        return vocab

    r"""
        Print the cupt file with the prediction in the Extended CoNLL-U format
    """

    def addPrediction(self, prediction):
        indexTokenPred = 0
        indexSentence = 0
        listTag = {}
        cpt = 0
        isVMWE = False
        for line in self.fileCupt:
            if self.isInASequence(line):
                newLine = ""
                if self.lineIsAComment(line):
                    print(line.split("\n")[0])
                else:
                    lineTMP = line.split("\t")
                    lineTMP[-1] = str(prediction[indexTokenPred])
                    tag, cpt, isVMWE = self.findTag(lineTMP, cpt, listTag, isVMWE)
                    indexTokenPred += 1
                    for ind in range(len(lineTMP) - 1):
                        newLine += str(lineTMP[ind]) + "\t"
                    print(newLine + tag)
            else:
                print(line.split("\n")[0])
                indexSentence += 1
                cpt = 0
                isVMWE = False
                listTag = {}

    r"""
        Find a tag in Extended CoNLL-U Format
    """

    def findTag(self, lineD, cpt, listTag, isVMWE):
        tag = lineD[-1]

        if tag == self.TagOuside or tag == "<unk>":  # or "-" in lineD[0] or "." in lineD[0]:
            tag = "*"
            isVMWE = False
            return tag, cpt, isVMWE

        if tag == self.TagGap:
            tag = "*"
            isVMWE = True
            return tag, cpt, isVMWE

        if tag[0] == self.TagBegin:
            isVMWE = True
            tag = tag[1:-1] + tag[-1]
            cpt += 1
            listTag[tag] = str(cpt)
            tag = str(cpt) + ":" + tag
            return tag, cpt, isVMWE

        if tag[0] == self.TagInside:
            tag = tag[1:-1] + tag[-1]
            if listTag.has_key(tag) and isVMWE:
                tag = listTag.get(tag)
            else:
                isVMWE = True
                cpt += 1
                listTag[tag] = str(cpt)
                tag = str(cpt) + ":" + tag
            return tag, cpt, isVMWE

        sys.stderr.write("Error with tags predict : {0} \n".format(tag))
        exit(1)

    r"""
        Return a dict with different VWME overlaps. 
    """

    def VMOverlaps(self, sequenceCupt):
        dictOverlaps = dict()
        dictVMWEvue = dict()
        numberVMWE = self.numberVMWEinSequence(sequenceCupt)
        indexDict = 0
        for index in range(numberVMWE):
            dictOverlaps[indexDict] = dict()
            for sequence in sequenceCupt:
                if len(sequence[-1].split(";")[index % len(sequence[-1].split(";"))].split(":")) > 1:
                    indexMWE = sequence[-1].split(";")[index % len(sequence[-1].split(";"))].split(":")[0]
                    MWE = sequence[-1].split(";")[index % len(sequence[-1].split(";"))].split(":")[1]
                    if "-" in sequence[0] and "." in sequence[0]:
                        continue

                    if not dictOverlaps[indexDict].has_key(indexMWE) and not dictVMWEvue.has_key(MWE):
                        if not self.endVMWE(int(sequence[0]), sequence, dictOverlaps[indexDict]):
                            dictOverlaps[indexDict][indexMWE] = MWE
                            dictVMWEvue[MWE] = indexDict
                        else:
                            dictOverlaps[indexDict+1] = dict()
                            dictOverlaps[indexDict+1][indexMWE] = MWE
                            dictVMWEvue[MWE] = indexDict + 1
            indexDict += 1

        return dictOverlaps

    r"""
        Create a Sequence with overlaps in the new format.
    """

    def createSequenceWithOverlaps(self, sequenceCupt):
        startVMWE = False
        comptUselessID = 1
        sequences = []
        dictOverlaps = self.VMOverlaps(sequenceCupt)

        for index in range(len(dictOverlaps)):
            listVMWE = {}  # self.createListSequence(sequenceCupt)
            for sequence in sequenceCupt:
                tagToken = ""
                tag = sequence[-1].split(";")[index % len(sequence[-1].split(";"))]
                if sequence[-1] != "*" and not "-" in sequence[0] and not "." in sequence[0]:
                    # update possible for many VMWE on one token
                    if len(tag.split(":")) > 1 and dictOverlaps[index].has_key(tag.split(":")[0]):
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

            if sequences not in self.resultSequences:
                self.resultSequences.append(sequences)
