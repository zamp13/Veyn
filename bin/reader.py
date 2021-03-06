#! /usr/bin/env python
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
        self.numberOfColumns = 0
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
                if self.numberOfColumns == 0:
                    if self.columnOfTags not in range(len(line.rstrip().split("\t"))) and self.test:
                        self.isConll = True
                        sys.stderr.write("Warning file: you test without tags column.\n")
                        self.numberOfColumns = len(line.rstrip().split("\t")) + 1
                    elif self.columnOfTags not in range(len(line.rstrip().split("\t"))) and not self.test:
                        sys.stderr.write("Error file: you can't train without column of tags.\n")
                        exit(404)
                    else:
                        self.numberOfColumns = len(line.rstrip().split("\t"))

                if self.isConll:
                    Newline = line.rstrip().split("\t")
                    for i in range(self.columnOfTags - len(Newline) + 1):
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
                elif tag in listVMWE:
                    indexVMWE = listVMWE.get(tag).split(":")[0]
                    if self.withVMWE:
                        VMWE = listVMWE.get(tag).split(":")[1]
                    else:
                        VMWE = ""
                    tagToken += self.TagInside + VMWE  # + "\t" + indexVMWE
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
            # if sequence[2] == "_":
            #    sequence[2] = sequence[1]
            # UPOS == _
            # if sequence[3] == "_":
            #    sequence[3] = sequence[4]

            newSequence = ""
            for index in range(len(sequence)):
                if index == self.columnOfTags:
                    newSequence += tagToken + "\t"
                else:
                    newSequence += sequence[index] + "\t"

            sequences.append(newSequence)
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
            if tag.split(":")[0] in listVWME:
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
        file = open(nameFileVocab, "w", encoding="utf-8")
        for key, voc in vocab.items():
            for keyToken, valVoc in voc.items():
                file.write(str(keyToken) + "\t" + str(valVoc) + "\n")
            file.write("\n")

    r"""
        Verify if words in resultSequences is in the vocab. 
    """

    def verifyUnknowWord(self, vocab):
        number = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for sentence in self.resultSequences:

            for line in range(len(sentence)):
                if isInASequence(sentence[line]):
                    lineTMP = sentence[line].rsplit("\t")
                    # flag = False
                    for col in range(self.numberOfColumns):
                        if lineTMP[col] not in vocab[col]:
                            # sys.stderr.write(str(sentence[line]) + "\n")
                            lineTMP[col] = "<unk>"
                            number[col] += 1
                            # flag = True

                    newLine = ""
                    for index in range(self.numberOfColumns):
                        newLine += lineTMP[index] + "\t"
                    sentence[line] = newLine
                    # if flag:
                    #    sys.stderr.write(str(sentence[line]) + "\n")
        print(number, file=sys.stderr)

    r"""
        Print the cupt file with the prediction in the Extended CoNLL-U format
    """

    def addPrediction(self, prediction, listNbToken):
        listTag = {}
        cpt = 0
        isVMWE = False
        error_prediction = False
        if self.numberOfSentence != len(prediction):
            print("Error number of sentence different of number sentence predict", file=sys.stderr)
            exit(300)

        for indexSentence in range(self.numberOfSentence):
            if error_prediction:
                for i in range(len(prediction[indexSentence - 1])):
                    p = self.resultSequences[indexSentence - 1][i].split("\t")
                    print(str(p[:4]), str(p[-2]), str(prediction[indexSentence - 1][i]), file=sys.stderr)
                print("\n", file=sys.stderr)
                error_prediction = False
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
                                #if len(lineTMP) < 11:
                                #    print(lineTMP, file=sys.stderr)
                                #    for i in range(11 - len(lineTMP)):
                                #        lineTMP.append("_")

                                lineTMP[self.columnOfTags] = str(prediction[indexSentence][indexPred])
                                tag, cpt, isVMWE, error_prediction = self.findTag(lineTMP, cpt, listTag, isVMWE, error_prediction)
                            else:
                                strError = "Warning: Error tags prediction! Sentence :" + str(
                                    indexSentence) + ",NbPrediction = " + str(
                                    len(prediction)) + ",NbPredictionSentence = " + str(
                                    len(prediction[indexSentence])) + ",Token ID want to predict : " + str(
                                    indexPred) + "\n"
                                sys.stderr.write(strError)

                        for ind in range(len(lineTMP)):
                            if ind == self.columnOfTags:
                                if ind == len(lineTMP) - 1:
                                    newLine += tag
                                else:
                                    newLine += tag + "\t"
                            else:
                                if ind == len(lineTMP) - 1:
                                    newLine += str(lineTMP[ind])
                                else:
                                    newLine += str(lineTMP[ind]) + "\t"
                        newSequence.append(newLine)
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

    def findTag(self, lineD, cpt, listTag, isVMWE, error_prediction):
        tag = lineD[self.columnOfTags]
        if self.withVMWE:
            if tag == self.TagOuside or tag == "<unk>":  # or "-" in lineD[0] or "." in lineD[0]:
                tag = "*"
                isVMWE = False
                return tag, cpt, isVMWE, error_prediction

            if tag == self.TagGap:
                tag = "*"
                isVMWE = True
                return tag, cpt, isVMWE, error_prediction

            if tag[0] == self.TagBegin:
                isVMWE = True
                tag = tag[1:-1] + tag[-1]
                cpt += 1
                listTag[tag] = str(cpt)
                tag = str(cpt) + ":" + tag
                return tag, cpt, isVMWE, error_prediction

            if tag[0] == self.TagInside:
                tag = tag[1:-1] + tag[-1]
                if tag in listTag and isVMWE:
                    tag = listTag.get(tag)
                else:
                    print("Warning: a tag I-", str(tag), " predicted without B-", str(tag), " before.", file=sys.stderr)
                    isVMWE = True
                    cpt += 1
                    listTag[tag] = str(cpt)
                    tag = str(cpt) + ":" + tag
                    error_prediction = True
                return tag, cpt, isVMWE, error_prediction

            sys.stderr.write("Error with tags predict : {0} \n".format(tag))
            exit(1)

        else:
            if tag == self.TagOuside or tag == "<unk>":  # or "-" in lineD[0] or "." in lineD[0]:
                tag = "*"
                isVMWE = False
                return tag, cpt, isVMWE, error_prediction

            if tag == self.TagGap:
                tag = "*"
                isVMWE = True
                return tag, cpt, isVMWE, error_prediction

            if tag[0] == self.TagBegin:
                isVMWE = True
                cpt += 1
                listTag[cpt] = str(cpt)
                tag = str(cpt) + ":MWE"
                return tag, cpt, isVMWE, error_prediction

            if tag[0] == self.TagInside:
                if cpt in listTag and isVMWE:
                    tag = listTag.get(cpt)
                else:
                    print("Warning: a tag ", str(tag), " predicted without B before.", file=sys.stderr)
                    isVMWE = True
                    cpt += 1
                    listTag[cpt] = str(cpt)
                    tag = str(cpt) + ":MWE"
                    error_prediction = True
                return tag, cpt, isVMWE, error_prediction

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
                if len(sequence[self.columnOfTags].split(";")[
                           index % len(sequence[self.columnOfTags].split(";"))].split(":")) > 1:
                    indexMWE = \
                        sequence[self.columnOfTags].split(";")[
                            index % len(sequence[self.columnOfTags].split(";"))].split(
                            ":")[0]
                    MWE = \
                        sequence[self.columnOfTags].split(";")[
                            index % len(sequence[self.columnOfTags].split(";"))].split(
                            ":")[1]
                    if "-" in sequence[0] and "." in sequence[0]:
                        continue

                    if indexMWE not in dictOverlaps[indexDict] and MWE not in dictVMWEvue:
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
        # sequences = []
        numberVMWE = self.numberVMWEinSequence(sequenceCupt)

        for index in range(numberVMWE):
            sequences = []
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
                    elif tag in listVMWE:
                        indexVMWE = listVMWE.get(tag).split(":")[0]
                        VMWE = listVMWE.get(tag).split(":")[1]
                        tagToken += self.TagInside + VMWE  # + "\t" + indexVMWE
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
                # if sequence[2] == "_":
                #    sequence[2] = sequence[1]
                # UPOS == _
                # if sequence[3] == "_":
                #    sequence[3] = sequence[4]

                newSequence = ""
                for index in range(len(sequence)):
                    if index == self.columnOfTags:
                        newSequence += tagToken + "\t"
                    else:
                        newSequence += sequence[index] + "\t"

                sequences.append(newSequence)
            sequences.append("\n")

            if sequences not in self.resultSequences:
                self.resultSequences.append(sequences)

    r"""Print the file cupt on the standard output"""

    def printFileCupt(self):
        for indexSentence in range(self.numberOfSentence):
            sentence = self.fileCupt[indexSentence]
            for line in sentence:
                print(line)

    r"""Print the result sentence on the standard output"""

    def printResultSequence(self):
        for sequence in self.resultSequences:
            for line in sequence:
                print(line)

    r"""Save the file cupt into file"""

    def saveFileCupt(self, file):
        for indexSentence in range(self.numberOfSentence):
            sentence = self.fileCupt[indexSentence]
            for line in sentence:
                print(line, file=file)

    r""" Construct sentences to train a fasttext models"""

    def construct_sentence(self, column):
        list_sentences_text = []

        for index_sentence in range(len(self.fileCupt)):
            list_sentences_text.append([])
            sentence = self.fileCupt[index_sentence]
            for line in sentence:
                if isInASequence(line) and not lineIsAComment(line) and "-" not in line.split("\t")[0]:
                    if len(line.split("\t")[column].split(" ")) > 1:
                        list_sentences_text[index_sentence].append(
                            line.split("\t")[column].split(" ")[0] + line.split("\t")[column].split(" ")[1])
                    else:
                        list_sentences_text[index_sentence].append(line.split("\t")[column])
        return list_sentences_text

    def petits_bateaux_pos_to_ud_pos(self):
        FORM = 1
        UPOS = 3
        for index_sentences in range(len(self.resultSequences)):
            for index_lines in range(len(self.resultSequences[index_sentences])):
                if self.resultSequences[index_sentences][index_lines] != "\n":
                    line = self.resultSequences[index_sentences][index_lines].split("\t")
                    if index_lines < len(self.resultSequences[index_sentences]) - 2:
                        line_next = self.resultSequences[index_sentences][index_lines+1].split("\t")
                        line[UPOS] = self.switch_pos(line[UPOS], line_next[UPOS], line[FORM])
                        new_line = ""
                        for feature in line:
                            new_line += feature + "\t"
                        self.resultSequences[index_sentences][index_lines] = new_line
                    else:
                        line[UPOS] = self.switch_pos(line[UPOS], "", line[FORM])
                        new_line = ""
                        for feature in line:
                            new_line += feature + "\t"
                        self.resultSequences[index_sentences][index_lines] = new_line

    def switch_pos(self, pos, next_pos, form):
        pos_vocab = {"det": "DET",
                     "nc": "NOUN",
                     "np": "PROPN",
                     "v": "VERB",
                     "vppart": "VERB",
                     "adj": "ADJ",
                     "adv": "ADV",
                     "prep": "ADP",
                     "coo": "CCONJ",
                     "csu": "SCONJ",
                     "clr": "PRON",
                     "cln": "PRON",
                     "clo": "PRON",
                     "prorel": "PRON"
                     }
        try:
            new_form = int(form)
            return "NUM"
        except Exception:
            pass
        if pos.startswith("ponct"):
            return "PUNCT"
        elif pos == "v" and next_pos == "vppart":
            return "AUX"
        elif pos.startswith("v"):
            return pos_vocab["v"]
        elif pos.startswith("adv"):
            return pos_vocab["adv"]
        elif pos.startswith("pro"):
            return pos_vocab["prorel"]
        elif pos.startswith("pri"):
            return pos_vocab["prorel"]
        elif pos.startswith("titre"):
            return pos_vocab["np"]
        elif pos.startswith("etr") or pos.startswith("X"):
            return pos_vocab["nc"]
        elif pos.startswith("pref") or pos.startswith("pres"):
            return pos_vocab["prep"]
        elif pos in pos_vocab:
            return pos_vocab[pos]

    def add_deprel_lemma(self):
        DEPREL = 6
        LEMMA = 2
        for index_sequence in range(len(self.resultSequences)):
            for index_line in range(len(self.resultSequences[index_sequence])):
                line = self.resultSequences[index_sequence][index_line]
                if line != "\n":
                    line = line.split("\t")
                    if "-" not in line[0] and "." not in line[0]:
                        # Replace by lemma linked
                        if line[DEPREL] == "0":
                            line[DEPREL] = "root"
                        else:
                            if line[DEPREL] != "_":
                                line[DEPREL] = self.resultSequences[index_sequence][int(line[DEPREL]) - 1].split("\t")[LEMMA]
                        # Reconstruction line
                        new_line = ""
                        for feat in range(len(line)):
                            new_line += line[feat] + "\t"
                        self.resultSequences[index_sequence][index_line] = new_line
