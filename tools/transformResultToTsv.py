#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function          
import fileinput
import sys
import re

def fileCompletelyRead(line):
    return line == ""
    
def lineIsAComment(line):
    return line[0] == "%"
    
def main():
    
    if(len(sys.argv) != 3):
        sys.stderr("Error, give a file with results (average and standard deviation) and an argument (MWE or TOKEN).\n")
        exit(1)
        
    filepathRes = sys.argv[1]
    measure = sys.argv[2]
    if(measure not in ["MWE", "TOKEN"]):
        sys.stderr("Wrong argument (MWE or TOKEN)\n")
        
    fileRes = open(filepathRes, "r")
    line = fileRes.readline()
    
    print("CorpusName\tP\tP_SD\tR\tR_SD\tF\tF_SD")
    corpusName = ""
    P = ""
    P_SD = ""
    R = ""
    R_SD = ""
    F = ""
    F_SD = ""
    
    while(not (fileCompletelyRead(line))):
        if(lineIsAComment(line)):
            corpusName = line[1:-1]
            line = fileRes.readline()
            line = line.strip().split(" ")
            if(line[0].startswith("Average")):
               corpusName += " on " + line[3] + " files"
               lineRes = fileRes.readline()
            if(measure == "TOKEN"):
                line = fileRes.readline()
                line = fileRes.readline()
                line = fileRes.readline()
                line = fileRes.readline()
                line = fileRes.readline()
            line = fileRes.readline()
            line = re.sub('[()]', '', line)
            line = line.split(" ")
            P = line[5]
            line = fileRes.readline()
            line = re.sub('[()]', '', line)
            line = line.split(" ")
            R = line[5]
            line = fileRes.readline()
            line = re.sub('[()]', '', line)
            line = line.split(" ")
            F = line[5][:-1]
            line = fileRes.readline()
            line = fileRes.readline()
            line = fileRes.readline()
            line = fileRes.readline()
            line = fileRes.readline()
            line = fileRes.readline()
            line = fileRes.readline()
            line = fileRes.readline()
            line = re.sub('[()]', '', line)
            line = line.split(" ")
            P_SD = line[5]
            line = fileRes.readline()
            line = re.sub('[()]', '', line)
            line = line.split(" ")
            R_SD = line[5]
            line = fileRes.readline()
            line = re.sub('[()]', '', line)
            line = line.split(" ")
            F_SD = line[5][:-1]
            
            print(corpusName+"\t"+P+"\t"+P_SD+"\t"+R+"\t"+R_SD+"\t"+F+"\t"+F_SD)
            line = fileRes.readline()
        else:
            line = fileRes.readline()
            continue
            
            
            
            
main()
