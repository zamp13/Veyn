#!/usr/bin/python
# -*- coding: utf-8 -*-
import fileinput
import sys
import re

def main():
    if(len(sys.argv) != 2):
        print("I need a file please! A file from a treeBank! Thanks! :)\n")
        exit(1)

    filepath = sys.argv[1]
    ftb      = open(filepath, "r")                #ouverture en lecture
    filename = filepath.split("/")[-1]
    filenamedimsum = re.sub('conll', 'dimsum', filename)
    fileDimsum = open(filenamedimsum, "w")        #ouverture en écriture ( fichier produit )
    flagBreak = False
    flagAddToken = False
    addToken = 0

    #lire tant qu'on ne trouve pas une ligne vide.
    lineFtb = ftb.readline()
    while(lineFtb != ""):
        while(lineFtb != "\n"):
            currentWordInformations = lineFtb.rstrip().split('\t')
            tokenOffset = int(currentWordInformations[0])
            word = currentWordInformations[1]
            lemma = currentWordInformations[2]
            POS = currentWordInformations[3]
            MWEinfos = currentWordInformations[5]
            
            if(word == '_'):
                word = ""
            if(lemma == '_'):
                lemma = ""        
            if(POS == '_'):
                POS = ""
                
            while("mwehead" in MWEinfos):
                #fileDimsum.write("%s\t%s\t%s\t%s\t%s\t%s\t\t\t_\n" % (tokenOffset, word, lemma, POS, "B", "0"))
                precTokenOffset = tokenOffset
                
                lineFtb = ftb.readline()
                if(lineFtb == "\n"):
                    fileDimsum.write("%s\t%s\t%s\t%s\t%s\t%s\t\t\t_\n" % (tokenOffset, word, lemma, POS, "O", "0"))
                    break
                nextWordInformations = lineFtb.rstrip().split('\t')
                nexttokenOffset = int(nextWordInformations[0])
                nextword = nextWordInformations[1]
                nextlemma = nextWordInformations[2]
                nextPOS = nextWordInformations[3]
                nextMWEinfos = nextWordInformations[5]
                
                if("pred=y" in nextMWEinfos and "mwehead" not in nextMWEinfos):
                    fileDimsum.write("%s\t%s\t%s\t%s\t%s\t%s\t\t\t_\n" % (tokenOffset, word, lemma, POS, "B", "0"))
                    fileDimsum.write("%s\t%s\t%s\t%s\t%s\t%s\t\t\t_\n" % (nexttokenOffset, nextword, nextlemma, nextPOS, "I", precTokenOffset))
                    precTokenOffset = nexttokenOffset
                    lineFtb = ftb.readline()
                    if(lineFtb == "\n"):
                        flagBreak = True
                        break
                else:
                    fileDimsum.write("%s\t%s\t%s\t%s\t%s\t%s\t\t\t_\n" % (tokenOffset, word, lemma, POS, "O", "0"))
                    fileDimsum.write("%s\t%s\t%s\t%s\t%s\t%s\t\t\t_\n" % (nexttokenOffset, nextword, nextlemma, nextPOS, "O", "0"))
                    lineFtb = ftb.readline()
                    if(lineFtb == "\n"):
                        break
                        
                currentWordInformations = lineFtb.rstrip().split('\t')
                tokenOffset = int(currentWordInformations[0])
                word = currentWordInformations[1]
                lemma = currentWordInformations[2]
                POS = currentWordInformations[3]
                MWEinfos = currentWordInformations[5]
                
                while("pred=y" in MWEinfos and "mwehead" not in MWEinfos):
                    fileDimsum.write("%s\t%s\t%s\t%s\t%s\t%s\t\t\t_\n" % (tokenOffset, word, lemma, POS, "I", precTokenOffset))
                    lineFtb = ftb.readline()
                    if(lineFtb == "\n"):
                        flagBreak = True
                        break
                    precTokenOffset = tokenOffset
                    currentWordInformations = lineFtb.rstrip().split('\t')
                    tokenOffset = int(currentWordInformations[0])
                    word = currentWordInformations[1]
                    lemma = currentWordInformations[2]
                    POS = currentWordInformations[3]
                    MWEinfos = currentWordInformations[5]
                if(flagBreak):
                    break
            if(flagBreak):
                flagBreak = False
                break
                
            fileDimsum.write("%s\t%s\t%s\t%s\t%s\t%s\t\t\t_\n" % (tokenOffset, word, lemma, POS, "O", "0"))
            lineFtb = ftb.readline()
        lineFtb = ftb.readline()
        fileDimsum.write("\n") 
main()




