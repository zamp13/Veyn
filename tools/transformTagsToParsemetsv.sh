#!/bin/bash

if [[ $# -ne 7 ]]; then
 echo -e "You need to give 7 arguments! \n- path file\n- nbLine to delete\n- first number file\n- lase number file\n- format (dimsum, parseme or blind)\n- gold format (.blind or dimsum16.test)\n- gold pasemetsv\n"
 exit 1
fi

FILE=$1
nbLineToDelete=$2
firstNumberFile=$3
lastNumberFile=$4
format=$5
goldFormat=$6
goldParsemetsv=$7



./multipleFilesKeepOnlyTags.sh $FILE $nbLineToDelete $firstNumberFile $lastNumberFile

for (( i=$firstNumberFile; i<=$lastNumberFile; i++ ))
do
    case "$format" in
    	"dimsum")
    	    ./addPredictToDimsumFile.py $goldFormat $FILE$i.txt > $FILE$i.dimsum
    	    ./dimsum_withGaps_ToParsemetsv.py $FILE$i.dimsum > $FILE$i.parsemetsv
    	    ;;
    	"parseme")
    	    ./addPredictParsemeFormat.py $goldFormat $FILE$i.txt > $FILE$i.parsemetsv 
    	    ;;
    	"blind")
    	    ./addPredictToBlindParsemetsv.py $goldFormat $FILE$i.txt > "$FILE$i".parsemetsv 
    	    ;;
    esac

    ../../data/sharedtask-data/bin/evaluate.py $goldParsemetsv $FILE$i.parsemetsv > tmp
    cat $FILE$i.parsemetsv >> tmp
    mv tmp $FILE$i.parsemetsv    
done

./standardDeviationAndAverage.py $FILE $lastNumberFile
