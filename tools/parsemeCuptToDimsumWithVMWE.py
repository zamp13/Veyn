#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import sys

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


class Main():

    def __init__(self, args):
        self.args = args

    def run(self):

        cupt = self.args.fileCupt

        line = cupt.readline()
        while (not self.fileCompletelyRead(line)):
            sequenceCupt = []
            while (self.isInASequence(line)):
                while (self.lineIsAComment(line)):
                    line = cupt.readline()
                sequenceCupt.append(line.rstrip().split("\t"))
                line = cupt.readline()
            self.createSequenceIO(sequenceCupt)
            # return                 ########################################## <----
            print
            line = cupt.readline()

    def fileCompletelyRead(self, line):
        return line == ""

    def isInASequence(self, line):
        return line != "\n" and line != ""

    def lineIsAComment(self, line):
        return line[0] == "#"

    def createSequenceIO(self, sequenceCupt):
        startVMWE = False
        listVMWE = {}  # self.createListSequence(sequenceCupt)
        comptUselessID = 1
        for sequence in sequenceCupt:
            tagToken = ""

            if sequence[-1] != "*" and len(sequence[-1].split(";")[0].split(":")) > 1:
                # update possible for many VMWE on one token
                tag = sequence[-1].split(";")[0]
                indexVMWE = tag.split(":")[0]
                VMWE = tag.split(":")[1]
                listVMWE[indexVMWE] = sequence[0] + ":" + VMWE
                tagToken += "I" + VMWE + "\t0"

            elif sequence[-1] != "*":
                tag = sequence[-1].split(";")[0]
                if listVMWE.has_key(tag):
                    indexVMWE = listVMWE.get(tag).split(":")[0]
                    VMWE = listVMWE.get(tag).split(":")[1]
                    tagToken += "I" + VMWE + "\t" + indexVMWE
                elif self.endVMWE(int(sequence[0])+comptUselessID, sequenceCupt, listVMWE):
                    tagToken += "o\t0"
                else:
                    tagToken += "0\t0"

            elif startVMWE and sequence[-1] == "*":
                tagToken += "o\t0"

            elif not startVMWE and sequence[-1] == "*":
                tagToken += "0\t0"
            if "-" in sequence[0] or "."in sequence[0]:
                comptUselessID += 1
            if not "-" in sequence[0] and not "."in sequence[0]:
                startVMWE = self.endVMWE(int(sequence[0])+comptUselessID, sequenceCupt, listVMWE)
            print(sequence[0] + "\t" + sequence[1] + "\t" + sequence[2] + "\t" + sequence[
                3] + "\t" + tagToken + "\t\t\t_")

    def endVMWE(self, param, sequenceCupt, listVWME):
        for index in range(param, len(sequenceCupt)):
            tag = sequenceCupt[index][-1].split(";")[0]

            if tag == "*":
                continue
            if listVWME.has_key(tag.split(":")[0]):
                return True
        return False


if __name__ == "__main__":
    Main(parser.parse_args()).run()
