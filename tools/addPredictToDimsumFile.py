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

def main():
    
    if(len(sys.argv) != 3):
        print("Error, give a dimsum file and a list of tags predicts for this file please.")
        exit(1)
        
    filepathDimsum = sys.argv[1]
    filepathTags = sys.argv[2]
    
    dimsum = open(filepathDimsum, "r")
    tags = open(filepathTags, "r")
    
    lineD = dimsum.readline()
    lineT = tags.readline()
    FlaglineTEmpty = False
    
    while(not (fileCompletelyRead(lineD) or fileCompletelyRead(lineT))):
        if(lineIsAComment(lineD)):
            print(lineD[:-1])
            lineD = dimsum.readline()
            continue
        if(lineIsAComment(lineT)):
            lineT = tags.readline()
            continue
            
        if(lineD == "\n" or lineT == "\n"):
            if(lineD == "\n" and lineT == "\n"):
                print()
                lineD = dimsum.readline()
                lineT = tags.readline()
                continue
            elif(lineT == "\n" and not (lineD == "\n")):
                FlaglineTEmpty = True
            else:
                print("------->",lineD, "-------->", lineT)
                sys.stderr.write("Error, two files not align?\n")
                exit(1)
                
        lineD = lineD[:-1].split("\t")
        newLine = ""
        continueNext = False
        
        for i in range(len(lineD)):
            if(continueNext):
                continueNext = False
                continue
            if(i == 4):
                if(FlaglineTEmpty):
                    newLine += "O\t0\t"
                else:
                    newLine += lineT[:-1]+"\t0\t"
                continueNext = True
            else:
                newLine += lineD[i]+"\t"
        
        print(newLine)
        
        lineD = dimsum.readline()
        if(not FlaglineTEmpty):
            lineT = tags.readline()
        FlaglineTEmpty = False
main()
