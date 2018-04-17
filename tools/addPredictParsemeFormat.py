#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function          
import fileinput
import sys
import re

def fileCompletelyRead(line):
    return line == ""
    
def isInASequence(line):
    return line != "\n" and line != ""
    
def lineIsAComment(line):
    return line[0] == "#"
    
def handleSequence(sequenceP, sequenceC):
    sequence = ""
    
    for i in range(len(sequenceP)):
        tokenP = sequenceP[i]
        tokenC = sequenceC[i]
        indexCurrentToken = tokenP[0]
        
        if("-" in indexCurrentToken): 
            continue
            
        if(tokenP[0] == tokenC[0] and tokenP[1] == tokenC[1]):
            tokenLine = tokenP[0]+"\t"+tokenP[1]+"\t"+tokenC[2]+"\t"+tokenC[3]+"\t"+tokenP[3]+"\n"
        else:
            sys.stderr.write("Error when read a sequence.")
            exit(1)
        sequence = sequence + tokenLine
        
    print(sequence) # add an empty line between each sequence

def main():
    
    if(len(sys.argv) != 3):
        print("Error, give a file with .parsemetsv format and a file with predict tags please.")
        exit(1)
        
    fileParsemetsv = sys.argv[1]
    filetags = sys.argv[2]
    
    parsemetsv = open(fileParsemetsv, "r")
    tags = open(filetags, "r")
    
    lineP = parsemetsv.readline()
    lineT = tags.readline()
    
    while(not (fileCompletelyRead(lineP) or fileCompletelyRead(lineP))):
        while(lineIsAComment(lineP)):
            lineP = parsemetsv.readline()
        while(lineIsAComment(lineT)):
            lineT = tags.readline()
        if("-" in lineP.split("\t")[0]):
            print(lineP[:-1])
            lineP = parsemetsv.readline()
            continue
        if((not isInASequence(lineT)) and (not isInASequence(lineP))):
            print()
        elif(isInASequence(lineT) and isInASequence(lineP)):
            lineP = lineP.split("\t")
            lineT = lineT[:-1]
            newLine = lineP[0]+"\t"+lineP[1]+"\t"+lineP[2]+"\t"+lineT
            print(newLine)
        else:        
            sys.stderr.write("error, not align\n")
            exit()
            
        lineP = parsemetsv.readline()
        lineT = tags.readline()
    
    
main()
