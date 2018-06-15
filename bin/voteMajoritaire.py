#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function          
import fileinput
import sys
import re
import os
import numpy as np

# python voteMajoritaire.py experiences/RNN/FR/tags/ > experiences/RNN/FR/tagsResultVoteMajoritaire.txt


existingTags = []

def fileCompletelyRead(line):
    return line == ""
    
def isInASequence(line):
    return line != "\n" and line != ""
    
def lineIsAComment(line):
    return line[0] == "#"

def collectTagsInOneFile(filename):
    tagsBioCurFile = []
    
    file = open(filename, "r")
    
    line = file.readline()
    while(not (fileCompletelyRead(line))):
        if(line.strip() not in existingTags):
            existingTags.append(line.strip())
        tagsBioCurFile.append(line.strip())
        line = file.readline()
    
    return tagsBioCurFile
    
def initChoice():
    choice = dict()
    
    for tag in existingTags:
        choice[tag] = 0
    return choice
    
def main():
    if(len(sys.argv) != 2):
        print("Error, give a directory (ending with /), containing the tags which you want to make the \"vote majoritaire\"")
        exit(1)
        
    path = sys.argv[1]
    
    tableTags = []
    
    for filename in os.listdir(path):
        tagsInOneFile = collectTagsInOneFile(path+filename)
        tableTags.append(tagsInOneFile)
    
    tagsFinal = []
    #print(tableTags, file=sys.stderr)
    for j in range(len(tableTags[0])):
        choice = initChoice()
        if(tableTags[0][j] == ""):
            print()
            continue
        for i in range(len(tableTags)):
            curTag = tableTags[i][j]
            choice[curTag] += 1
        if(choice["O"] >= len(tableTags)/2): # len(tabletags) corresponds to the number of files
            print("O")
        else:
            bestTag = ""
            bestScore = 0
            for tag, nbCurTag in choice.iteritems():
                if(tag == "O"):
                    continue
                if(nbCurTag > bestScore):
                    bestTag = tag
                    bestScore = nbCurTag
            print(bestTag)
            
    
    # for tags in tableTags:
    #     curString = ""
    #     print(tags[0])
    #     for i in range(len(tags)):
    #         curString += str(tags[i]) + " "
    #     print(curString)
main()
