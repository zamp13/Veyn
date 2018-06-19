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


r"""
        Return if a file is completely read.
"""


def fileCompletelyRead(line):
    return line == ""


r"""
    Return if is not the and of the sentence.
"""


def isInASequence(line):
    return line != "\n" and line != ""


r"""
    Return if the line is a comment.
"""


def lineIsAComment(line):
    return line[0] == "#"


class ReaderCupt:

    def __init__(self, FORMAT, withOverlaps, test, file, columnOfTags):
        self.file = file
        self.columnOfTags = columnOfTags
        self.fileCupt = {}
        self.numberOfSentence = 0
        self.resultSequences = []
        self.TagBegin = "0"
        self.TagInside = "0"
        self.TagOuside = "0"
        self.TagGap = "0"
        self.withVMWE = False
        self.withOverlaps = withOverlaps
        self.FORMAT = FORMAT
        self.test = test
        self.isConll = False
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
        Read and transform in the good format in [BIO, BIOg, BIOcat, BIOgcat, IO, IOg, IOcat, IOgcat].
    """

    def read(self):
        line = self.file.readline()
        while not fileCompletelyRead(line):
            sequenceCupt = []
            sequenceFileCupt = []
            while isInASequence(line):
                while lineIsAComment(line):
                    sequenceFileCupt.append(line.split("\n")[0])
                    line = self.file.readline()
                if len(line.rstrip().split("\t")) < 11 and self.test:
                    self.isConll = True
                elif len(line.rstrip().split("\t")) < 10 and not self.test:
                    sys.stderr.write("Error file: you can't train on ConLL file.\n")

                if self.isConll:
                    Newline = line.rstrip().split("\t")
                    Newline.append("_")
                    sequenceCupt.append(Newline)
                    sequenceFileCupt.append(line.split("\n")[0] + "\t_")
                else:
                    sequenceCupt.append(line.rstrip().split("\t"))
                    sequenceFileCupt.append(line.split("\n")[0])
                line = self.file.readline()
            if self.withOverlaps and not self.test:
                self.createSequenceWithOverlaps(sequenceCupt)
            else:
                self.createSequence(sequenceCupt)

            sequenceFileCupt.append("")
            line = self.file.readline()
            self.fileCupt[self.numberOfSentence] = sequenceFileCupt
            self.numberOfSentence += 1

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
            tag = sequence[self.columnOfTags].split(";")[0]
            if sequence[self.columnOfTags] != "*" and not "-" in sequence[0] and not "." in sequence[0]:
                # update possible for many VMWE on one token
                if len(tag.split(":")) > 1:
                    indexVMWE = tag.split(":")[0]
                    VMWE = tag.split(":")[1]
                    listVMWE[indexVMWE] = sequence[0] + ":" + VMWE
                    if self.withVMWE:
                        tagToken += self.TagBegin + VMWE
                    else:
                        tagToken += self.TagBegin
                elif listVMWE.has_key(tag):
                    indexVMWE = listVMWE.get(tag).split(":")[0]
                    if self.withVMWE:
                        VMWE = listVMWE.get(tag).split(":")[1]
                    else:
                        VMWE = ""
                    tagToken += self.TagInside + VMWE #+ "\t" + indexVMWE
                elif self.endVMWE(int(sequence[0]) + comptUselessID, sequenceCupt, listVMWE):
                    tagToken += self.TagGap
                else:
                    tagToken += self.TagOuside

            elif startVMWE and sequence[self.columnOfTags] == "*":
                tagToken += self.TagGap
            elif not startVMWE and sequence[self.columnOfTags] == "*" or sequence[self.columnOfTags] == "_":
                tagToken += self.TagOuside

            if "-" in sequence[0] or "." in sequence[0]:
                comptUselessID += 1
            if not "-" in sequence[0] and not "." in sequence[0]:
                startVMWE = self.endVMWE(int(sequence[0]) + comptUselessID, sequenceCupt, listVMWE)

            # Lemma == _
            if sequence[2] == "_":
                sequence[2] = sequence[1]
            # UPOS == _
            if sequence[3] == "_":
                sequence[3] = sequence[4]

            newSequence = ""
            for index in range(len(sequence) - 1):
                newSequence += sequence[index] + "\t"

            sequences.append(newSequence + tagToken)
        sequences.append("\n")
        self.resultSequences.append(sequences)

    r"""
        Verify if the end of overlaps 
    """

    def endVMWE(self, param, sequenceCupt, listVWME):
        for index in range(param, len(sequenceCupt)):
            tag = sequenceCupt[index][self.columnOfTags].split(";")[0]

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
            if sequence[self.columnOfTags] == "*" or sequence[self.columnOfTags] == "_":
                continue

            if len(sequence[self.columnOfTags].split(";")) > numberVMWE:
                numberVMWE = len(sequence[self.columnOfTags].split(";"))
        return numberVMWE

    r"""
        Save vocab dict into nameFileVocab
    """

    def saveVocab(self, nameFileVocab, vocab):
        file = open(nameFileVocab, "w")
        for key, voc in vocab.items():
            for keyToken, valVoc in voc.items():
                file.write(str(keyToken) + "\t" + str(valVoc) + "\n")
            file.write("\n")

    r"""
        Verify if words in resultSequences is in the vocab. 
    """

    def verifyUnknowWord(self, vocab):
        for sentence in self.resultSequences:

            for line in range(len(sentence)):
                if isInASequence(sentence[line]):
                    lineTMP = sentence[line].split("\t")
                    #flag = False
                    for col in range(len(lineTMP)):
                        if not vocab[col].has_key(lineTMP[col]):
                            #sys.stderr.write(str(sentence[line]) + "\n")
                            lineTMP[col] = "<unk>"
                            #flag = True

                    newLine = ""
                    for index in range(len(lineTMP)):
                        newLine += lineTMP[index] + "\t"

                    sentence[line] = newLine
                    #if flag:
                    #    sys.stderr.write(str(sentence[line]) + "\n")

    r"""
        Print the cupt file with the prediction in the Extended CoNLL-U format
    """

    def addPrediction(self, prediction, listNbToken):
        listTag = {}
        cpt = 0
        isVMWE = False

        for indexSentence in range(self.numberOfSentence):
            sentence = self.fileCupt[indexSentence]
            newSequence = []
            indexPred = 0
            for indexLine in range(len(sentence)):
                sequence = sentence[indexLine]
                if isInASequence(sequence):
                    newLine = ""
                    if not lineIsAComment(sequence):
                        lineTMP = sequence.split("\t")

                        tag = "*"
                        if not "-" in lineTMP[0] and not "." in lineTMP[0]:

                            if indexPred < len(prediction[indexSentence]):
                                lineTMP[self.columnOfTags] = str(prediction[indexSentence][indexPred])
                                tag, cpt, isVMWE = self.findTag(lineTMP, cpt, listTag, isVMWE)
                            else:
                                strError = "Warning: Error tags prediction! Sentence :" + str(
                                    indexSentence) + ",NbPrediction = " + str(
                                    len(prediction)) + ",NbPredictionSentence = " + str(
                                    len(prediction[indexSentence])) + ",Token ID want to predict : " + str(
                                    indexPred) + "\n"
                                sys.stderr.write(strError)

                        for ind in range(len(lineTMP) - 1):
                            newLine += str(lineTMP[ind]) + "\t"
                        newSequence.append(newLine + tag)
                        indexPred += 1
                    else:
                        newSequence.append(sequence)
                else:
                    newSequence.append(sequence)
                    indexPred = 0
                    cpt = 0
                    isVMWE = False
                    listTag = {}
            self.fileCupt[indexSentence] = newSequence

    r"""
        Find a tag in Extended CoNLL-U Format
    """

    def findTag(self, lineD, cpt, listTag, isVMWE):
        tag = lineD[self.columnOfTags]

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
                if len(sequence[self.columnOfTags].split(";")[index % len(sequence[self.columnOfTags].split(";"))].split(":")) > 1:
                    indexMWE = sequence[self.columnOfTags].split(";")[index % len(sequence[self.columnOfTags].split(";"))].split(":")[0]
                    MWE = sequence[self.columnOfTags].split(";")[index % len(sequence[self.columnOfTags].split(";"))].split(":")[1]
                    if "-" in sequence[0] and "." in sequence[0]:
                        continue

                    if not dictOverlaps[indexDict].has_key(indexMWE) and not dictVMWEvue.has_key(MWE):
                        if not self.endVMWE(int(sequence[0]), sequence, dictOverlaps[indexDict]):
                            dictOverlaps[indexDict][indexMWE] = MWE
                            dictVMWEvue[MWE] = indexDict
                        else:
                            dictOverlaps[indexDict + 1] = dict()
                            dictOverlaps[indexDict + 1][indexMWE] = MWE
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
        numberVMWE = self.numberVMWEinSequence(sequenceCupt)

        for index in range(numberVMWE):
            listVMWE = {}  # self.createListSequence(sequenceCupt)
            for sequence in sequenceCupt:
                tagToken = ""
                tag = sequence[self.columnOfTags].split(";")[index % len(sequence[self.columnOfTags].split(";"))]
                if sequence[self.columnOfTags] != "*" and not "-" in sequence[0] and not "." in sequence[0]:
                    # update possible for many VMWE on one token
                    if len(tag.split(":")) > 1:
                        indexVMWE = tag.split(":")[0]
                        VMWE = tag.split(":")[1]
                        listVMWE[indexVMWE] = sequence[0] + ":" + VMWE
                        tagToken += self.TagBegin + VMWE
                    elif listVMWE.has_key(tag):
                        indexVMWE = listVMWE.get(tag).split(":")[0]
                        VMWE = listVMWE.get(tag).split(":")[1]
                        tagToken += self.TagInside + VMWE# + "\t" + indexVMWE
                    elif self.endVMWE(int(sequence[0]) + comptUselessID, sequenceCupt, listVMWE):
                        tagToken += self.TagGap
                    else:
                        tagToken += self.TagOuside

                elif startVMWE and sequence[self.columnOfTags] == "*":
                    tagToken += self.TagGap
                elif not startVMWE and sequence[self.columnOfTags] == "*" or sequence[self.columnOfTags] == "_":
                    tagToken += self.TagOuside

                if "-" in sequence[0] or "." in sequence[0]:
                    comptUselessID += 1
                if not "-" in sequence[0] and not "." in sequence[0]:
                    startVMWE = self.endVMWE(int(sequence[0]) + comptUselessID, sequenceCupt, listVMWE)

                # Lemma == _
                if sequence[2] == "_":
                    sequence[2] = sequence[1]
                # UPOS == _
                if sequence[3] == "_":
                    sequence[3] = sequence[4]

                newSequence = ""
                for index in range(len(sequence) - 1):
                    newSequence += sequence[index] + "\t"

                sequences.append(newSequence + tagToken)
            sequences.append("\n")

            if sequences not in self.resultSequences:
                self.resultSequences.append(sequences)

    def printFileCupt(self):
        for indexSentence in range(self.numberOfSentence):
            sentence = self.fileCupt[indexSentence]
            for line in sentence:
                print(line)

    def printResultSequence(self):
        for sequence in self.resultSequences:
            for line in sequence:
                print(line)