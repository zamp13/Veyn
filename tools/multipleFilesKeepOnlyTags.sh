#!/bin/bash

FILE=$1
nbLineToDelete=$2
firstNumberFile=$3
lastNumberFile=$4

for i in `seq $3 $4`;
do
    tail -n +"$2" "$FILE$i.txt" > "$FILE.tmp" && mv "$FILE.tmp" "$FILE$i.txt"
done
