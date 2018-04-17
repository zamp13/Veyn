#!/bin/bash

train=$1
test=$2
dimTrain=$3
dimTest=$4
fileResult=$5

# .cupt --> .dimsum
./parsemeFileToDimsum.py "$train" > "$dimTrain"
./parsemeFileToDimsum.py "$test" > "$dimTest"

# train and predict
./RNNMultiGRU.py --ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4 --train="$dimTrain" --test="$dimTest" > "$fileResult"

# add predict te the .dimsum test --> predict_$dimTest
./addPredictToDimsumFile.py "$dimTest" "$fileResult" > "predict_$dimTest"

# .dimsum --> .cupt (predict_$test)
./dimsum_withGaps_ToCupt.py "$test" "predict_$dimTest" > "predict_$test"