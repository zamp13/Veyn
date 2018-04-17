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
    
def findTag(line, cpt, previousTag):
    tag = line[4]
    if(tag.upper() == "O"):
        return "_", cpt
    if(tag == "B"):
        cpt += 1
        tag = str(cpt)+":CRF"
        return tag, cpt
    if(tag == "I"):
        if(previousTag == "_"):
            cpt += 1
            tag = str(cpt)+":CRF"
            return tag, cpt
        else:
            return str(cpt), cpt
    sys.stderr.write("Error with tags predict : {0} \n".format(tag))
    exit(1)
    
    
def main():
    
    if(len(sys.argv) != 2):
        print("Error, give a dimsum file.")
        exit(1)
        
    filepathDimsum = sys.argv[1]
    
    fileDimsum = open(filepathDimsum, "r")
    
    line = fileDimsum.readline()
    previousTag = "_"
    cpt = 0
    
    while(not (fileCompletelyRead(line))):
        if(lineIsAComment(line)):
            continue
        if(line == "\n"):
            print()
            line = fileDimsum.readline()
            cpt = 0
            previousTag = "_"
            continue
            
        line = line.split("\t")
        tag, cpt = findTag(line, cpt, previousTag)
        newLine = line[0]+"\t"+line[1]+"\t_\t"+tag
        print(newLine)
        previousTag = tag[0]
        line = fileDimsum.readline()
        
main()
