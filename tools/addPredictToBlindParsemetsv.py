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
    
def findTag(lineT, cpt, correspondingNumber, FlaglineTEmpty):
    if(FlaglineTEmpty):
        return "_", cpt, correspondingNumber
    tag = lineT[0]
    if(tag.upper() == "O"):
        return "_", cpt, correspondingNumber
    if(tag == "B"):
        cpt += 1
        correspondingNumber[lineT[1]] = cpt
        return str(cpt)+":CRF", cpt, correspondingNumber
    if(tag == "I"):
        ######## Considère qu'il y a toujours un B avant de voir un I
        if(lineT[1] in correspondingNumber):
            return str(correspondingNumber[lineT[1]]), cpt, correspondingNumber
        else:
            cpt += 1
        correspondingNumber[lineT[1]] = cpt
        return str(cpt)+":CRF", cpt, correspondingNumber
    sys.stderr.write("Error with tags predict : {0} \n".format(tag))
    exit(1)
    
    
def main():
    
    if(len(sys.argv) != 3):
        print("Error, give a file with .blind.parsemetsv format and a list of tags predicts please.")
        exit(1)
        
    filepathBlind = sys.argv[1]
    filepathTags = sys.argv[2]
    
    blind = open(filepathBlind, "r")
    tags = open(filepathTags, "r")
    
    lineB = blind.readline()
    lineT = tags.readline()

    cpt = 0
    correspondingNumber = dict()
    FlaglineTEmpty = False
    
    while(not (fileCompletelyRead(lineB) or fileCompletelyRead(lineT))):
        if(lineIsAComment(lineB)):
            print(lineB[:-1])
            lineB = blind.readline()
            continue
        if(lineIsAComment(lineT)):
            lineT = tags.readline()
            continue
            
        if("-" in lineB.split("\t")[0]):
            print(lineB[:-1])
            lineB = blind.readline()
            continue
            
        if(lineB == "\n" or lineT == "\n"):
            if(lineB == "\n" and lineT == "\n"):
                print()
                lineB = blind.readline()
                lineT = tags.readline()
                cpt = 0
                correspondingNumber = dict()
                continue
            elif(lineT == "\n" and not (lineB == "\n")):
                FlaglineTEmpty = True
            else:
                print("------->",lineB, "-------->", lineT)
                sys.stderr.write("Error, two files not align?\n")
                exit(1)
                
        lineB = lineB[:-1]
        lineB = "\t".join(lineB.split("\t")[:-1])
        tag, cpt, correspondingNumber = findTag(lineT, cpt, correspondingNumber, FlaglineTEmpty)
        newLine = lineB+"\t"+tag
        print(newLine)
        
        lineB = blind.readline()
        if(not FlaglineTEmpty):
            lineT = tags.readline()
        FlaglineTEmpty = False
main()
