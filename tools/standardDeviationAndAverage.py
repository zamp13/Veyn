#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function          
import fileinput
import sys
import re
import os.path
import numpy as np

mweP = []
mweR = []
mweF = []
mweGoodAnnot = []
mweTotalAnnot = []
mweReal = []
tokenP = []
tokenR = []
tokenF = []
tokenGoodAnnot = []
tokenTotalAnnot = []
tokenReal = []
nbFiles = 0.0

def handle_file(file_path):
    global mweP, mweR, mweF, mweGoodAnnot, mweTotalAnnot, mweReal, tokenP, tokenR, tokenF, tokenGoodAnnot, tokenTotalAnnot, tokenReal, nbFiles
    nbFiles += 1
    curFile = open(file_path, "r")
    
    line = curFile.readline() 
    
    line = curFile.readline()
    line = re.sub('[()]', '', line)
    line = line.split(" ")
    mweP.append(float(line[5]))
    mweGoodAnnot.append(float(line[6]))
    mweTotalAnnot.append(float(line[8]))
    
    line = curFile.readline()
    line = re.sub('[()]', '', line)
    line = line.split(" ")
    mweR.append(float(line[5]))
    mweReal.append(float(line[8]))
    
    line = curFile.readline()
    line = line.split(" ")
    mweF.append(float(line[5]))
    
    line = curFile.readline()
    line = curFile.readline()
    
    line = curFile.readline()
    line = re.sub('[()]', '', line)
    line = line.split(" ")
    tokenP.append(float(line[5]))
    tokenGoodAnnot.append(float(line[6]))
    tokenTotalAnnot.append(float(line[8]))
    
    line = curFile.readline()
    line = re.sub('[()]', '', line)
    line = line.split(" ")
    tokenR.append(float(line[5]))
    tokenReal.append(float(line[8]))
    
    line = curFile.readline()
    line = line.split(" ")
    tokenF.append(float(line[5]))

def main():
    global mweP, mweR, mweF, mweGoodAnnot, mweTotalAnnot, mweReal, tokenP, tokenR, tokenF, tokenGoodAnnot, tokenTotalAnnot, tokenReal, nbFiles
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
            

    mweP = np.array(mweP)
    mweR = np.array(mweR)
    mweF = np.array(mweF)
    mweGoodAnnot = np.array(mweGoodAnnot)
    mweTotalAnnot = np.array(mweTotalAnnot)
    mweReal = np.array(mweReal)
    tokenP = np.array(tokenP)
    tokenR = np.array(tokenR)
    tokenF = np.array(tokenF)
    tokenGoodAnnot = np.array(tokenGoodAnnot)
    tokenTotalAnnot = np.array(tokenTotalAnnot)
    tokenReal = np.array(tokenReal)
    
    
    print("Average results on "+str(int(nbFiles))+" files:")
    print(">> MWE-based:")
    print("  * P = "+format((np.mean(mweP)*100), '.2f')+" ("+str(int(np.mean(mweGoodAnnot)))+" / "+str(int(np.mean(mweTotalAnnot)))+")")
    print("  * R = "+format((np.mean(mweR)*100), '.2f')+" ("+str(int(np.mean(mweGoodAnnot)))+" / "+str(int(np.mean(mweReal)))+")")
    print("  * F = "+format((np.mean(mweF)*100), '.2f'))
    print()
    print(">> Token-based:")
    print("  * P = "+format((np.mean(tokenP)*100), '.2f')+" ("+str(int(np.mean(tokenGoodAnnot)))+" / "+str(int(np.mean(tokenTotalAnnot)))+")")
    print("  * R = "+format((np.mean(tokenR)*100), '.2f')+" ("+str(int(np.mean(tokenGoodAnnot)))+" / "+str(int(np.mean(tokenReal)))+")")
    print("  * F = "+format((np.mean(tokenF)*100), '.2f'))

    print("Standard Deviation results on "+str(int(nbFiles))+" files:")
    print(">> MWE-based:")
    print("  * P = "+format((np.std(mweP)*100), '.2f')+" ("+str(int(np.std(mweGoodAnnot)))+" / "+str(int(np.std(mweTotalAnnot)))+")")
    print("  * R = "+format((np.std(mweR)*100), '.2f')+" ("+str(int(np.std(mweGoodAnnot)))+" / "+str(int(np.std(mweReal)))+")")
    print("  * F = "+format((np.std(mweF)*100), '.2f'))
    print()
    print(">> Token-based:")
    print("  * P = "+format((np.std(tokenP)*100), '.2f')+" ("+str(int(np.std(tokenGoodAnnot)))+" / "+str(int(np.std(tokenTotalAnnot)))+")")
    print("  * R = "+format((np.std(tokenR)*100), '.2f')+" ("+str(int(np.std(tokenGoodAnnot)))+" / "+str(int(np.std(tokenReal)))+")")
    print("  * F = "+format((np.std(tokenF)*100), '.2f'))
    
    
main()


# >> MWE-based:
#   * P = 0.2631 (186 / 707)
#   * R = 0.3720 (186 / 500)
#   * F = 0.3082

# >> Token-based:
#   * P = 0.4114 (636 / 1546)
#   * R = 0.5740 (636 / 1108)
#   * F = 0.4793
