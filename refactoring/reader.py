import sys

"""
    Reader class to read cupt file.
    Give a boolean test and a FORMAT in [BIO, BIOg, BIOcat, BIOgcat, IO, IOg, IOcat, IOgcat] to initialize it.
"""


class ReaderCupt:

    def __init__(self, FORMAT, test):
        self.FORMAT = FORMAT
        self.test = test

    def run(self, file):
        if self.test:
            return self.read_test(file)
        elif self.FORMAT == "BIO":
            return self.read_to_bio(file)
        elif self.FORMAT == "IO":
            return self.read_to_io(file)
        elif self.FORMAT == "BIOg":
            return self.read_to_bio_with_gap(file)
        elif self.FORMAT == "IOg":
            return self.read_to_io_with_gap(file)
        elif self.FORMAT == "BIOcat":
            return self.read_to_bio_with_cat(file)
        elif self.FORMAT == "IOcat":
            return self.read_to_io_with_cat(file)
        elif self.FORMAT == "BIOgcat":
            return self.read_to_bio_with_gap_and_cat(file)
        elif self.FORMAT == "IOgcat":
            return self.read_to_io_with_gap_and_cat(file)
        else:
            print("Error ", self.FORMAT, file=sys.stderr)

    def fileCompletelyRead(self, line):
        return line == ""

    def isInASequence(self, line):
        return line != "\n" and line != ""

    def lineIsAComment(self, line):
        return line[0] == "#"

    def read_test(self, file):
        line = file.readline()
        while (not self.fileCompletelyRead(line)):
            sequenceCupt = []
            while (self.isInASequence(line)):
                while (self.lineIsAComment(line)):
                    line = file.readline()
                sequenceCupt.append(line.rstrip().split("\t"))
                line = file.readline()
            self.createSequenceIO(sequenceCupt, self.test)
            line = file.readline()

    def read_to_bio(self, file):
        pass

    def read_to_io(self, file):
        pass

    def read_to_bio_with_gap(self, file):
        pass

    def read_to_io_with_gap(self, file):
        pass

    def read_to_bio_with_cat(self, file):
        pass

    def read_to_io_with_cat(self, file):
        pass

    def read_to_bio_with_gap_and_cat(self, file):
        pass

    def read_to_io_with_gap_and_cat(self, file):
        pass

    def createSequenceIO(self, sequenceCupt, test):
        startVMWE = False
        comptUselessID = 1

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
                        tagToken += "B" + VMWE + "\t0"
                    elif listVMWE.has_key(tag):
                        indexVMWE = listVMWE.get(tag).split(":")[0]
                        VMWE = listVMWE.get(tag).split(":")[1]
                        tagToken += "I" + VMWE + "\t" + indexVMWE
                    elif self.endVMWE(int(sequence[0]) + comptUselessID, sequenceCupt, listVMWE):
                        tagToken += "o\t0"
                    else:
                        tagToken += "O\t0"

                elif startVMWE and sequence[-1] == "*":
                    tagToken += "o\t0"
                elif not startVMWE and sequence[-1] == "*" or sequence[-1] == "_":
                    tagToken += "O\t0"

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

                print(newSequence + tagToken + "\t\t\t_")
            print

    def endVMWE(self, param, sequenceCupt, listVWME):
        for index in range(param, len(sequenceCupt)):
            tag = sequenceCupt[index][-1].split(";")[0]

            if tag == "*":
                continue
            if listVWME.has_key(tag.split(":")[0]):
                return True
        return False


    def numberVMWEinSequence(self, sequenceCupt):
        numberVMWE = 1
        for sequence in sequenceCupt:
            if sequence[-1] == "*" or sequence[-1] == "_":
                continue

            if len(sequence[-1].split(";")) > numberVMWE :
                numberVMWE = len(sequence[-1].split(";"))
        return numberVMWE