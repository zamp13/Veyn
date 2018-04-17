#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function          
import fileinput
import sys
import re
import os.path

mweP = mweR = mweF = mweGoodAnnot = mweTotalAnnot = mweReal = tokenP = tokenR = tokenF = tokenGoodAnnot = tokenTotalAnnot = tokenReal = nbFiles = 0.0

def handle_file(file_path):
    global mweP, mweR, mweF, mweGoodAnnot, mweTotalAnnot, mweReal, tokenP, tokenR, tokenF, tokenGoodAnnot, tokenTotalAnnot, tokenReal, nbFiles
    nbFiles += 1
    curFile = open(file_path, "r")
    
    line = curFile.readline() 
    
    line = curFile.readline()
    line = re.sub('[()]', '', line)
    line = line.split(" ")
    mweP += float(line[5])
    mweGoodAnnot += float(line[6])
    mweTotalAnnot += float(line[8])
    
    line = curFile.readline()
    line = re.sub('[()]', '', line)
    line = line.split(" ")
    mweR += float(line[5])
    mweReal += float(line[8])
    
    line = curFile.readline()
    line = line.split(" ")
    mweF += float(line[5])
    
    line = curFile.readline()
    line = curFile.readline()
    
    line = curFile.readline()
    line = re.sub('[()]', '', line)
    line = line.split(" ")
    tokenP += float(line[5])
    tokenGoodAnnot += float(line[6])
    tokenTotalAnnot += float(line[8])
    
    line = curFile.readline()
    line = re.sub('[()]', '', line)
    line = line.split(" ")
    tokenR += float(line[5])
    tokenReal += float(line[8])
    
    line = curFile.readline()
    line = line.split(" ")
    tokenF += float(line[5])

def main():
    nbArgs = len(sys.argv)-1
    if(nbArgs < 2):
        print("Error, give a path (or several) of file starting with results from eval script parsemetsv (eg. ../testBase will consider testBase.parsemetsv, testBase2.parsemetsv etc...) and the bigger number of these files.")
        exit(1)
        
    pathFiles = []
    
    for i in range(1, len(sys.argv)-1):
        pathFiles.append(sys.argv[i])
    maxFiles = int(sys.argv[len(sys.argv)-1])
    for pathFile in pathFiles:
        for i in range(1, maxFiles+1):
            file_path = pathFile+str(i)+".parsemetsv"
            if(not os.path.exists(file_path)):
                continue
            else:
                handle_file(file_path)
        file_path = pathFile+".parsemetsv"
        if(os.path.exists(file_path)):
            handle_file(file_path)
    
    print("Average results on "+str(int(nbFiles))+" files:")
    print(">> MWE-based:")
    print("  * P = "+format((mweP*100/nbFiles), '.2f')+" ("+str(int(mweGoodAnnot/nbFiles))+" / "+str(int(mweTotalAnnot/nbFiles))+")")
    print("  * R = "+format((mweR*100/nbFiles), '.2f')+" ("+str(int(mweGoodAnnot/nbFiles))+" / "+str(int(mweReal/nbFiles))+")")
    print("  * F = "+format((mweF*100/nbFiles), '.2f'))
    print()
    print(">> Token-based:")
    print("  * P = "+format((tokenP*100/nbFiles), '.2f')+" ("+str(int(tokenGoodAnnot/nbFiles))+" / "+str(int(tokenTotalAnnot/nbFiles))+")")
    print("  * R = "+format((tokenR*100/nbFiles), '.2f')+" ("+str(int(tokenGoodAnnot/nbFiles))+" / "+str(int(tokenReal/nbFiles))+")")
    print("  * F = "+format((tokenF*100/nbFiles), '.2f'))
    
main()


# >> MWE-based:
#   * P = 0.2631 (186 / 707)
#   * R = 0.3720 (186 / 500)
#   * F = 0.3082

# >> Token-based:
#   * P = 0.4114 (636 / 1546)
#   * R = 0.5740 (636 / 1108)
#   * F = 0.4793
