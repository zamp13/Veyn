#!/bin/bash

DATA="../data/sharedtask-data/1.1/"
RESULT="../result/BIOVMWE/"

LANG=${1}"/"
MODELS="../Models/"${LANG}"model1.h5"
# BG/ DE/ EL/ EN/ ES/ EU/ FA/ FR/ HE/ HI/ HR/ HU/ IT/ LT/ PL/ PT/ RO/ SL/ TR/
TRAIN="train"
CUPT=".cupt"
PRED=".pred"
DIMSUM=".dimsum"
TEST="test.blind"
TEST_RESULT="test.system"
PREDICT="predictBIOvmwe-"
OPT_COLUMNS="--ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4"
OPT_TRAIN=" --train="
OPT_TEST=" --test="
OPT_TEST_BIS=" --test "
OPT_CUPT=" --cupt "
OPT_CUPT=" --cupt "
OPT_DIMSUM=" --dimsum "
OPT_TAG=" --tag "

# .cupt --> .dimsum
echo "Start parse "${CUPT}" to "${DIMSUM}"."
echo ${DATA}${LANG}${TEST}${CUPT}
./parsemeCuptToDimsumWithBIOVMWE.py ${OPT_CUPT} ${DATA}${LANG}${TEST}${CUPT} ${OPT_TEST_BIS} > ${DATA}${LANG}${TEST}${DIMSUM}
echo "End parse "${CUPT}" to "${DIMSUM}"."
# train and predict
#./RNNMultiGRUWithVMWE.py --ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4 --train="$dimTrain" --test="$dimTest" > "$fileResult"
echo "train = "${DATA}${LANG}${TRAIN}${DIMSUM}" and test = "${DATA}${LANG}${TEST}${DIMSUM}
ARGUMENTS=${OPT_COLUMNS}${OPT_TRAIN}${MODELS}${OPT_TEST}${DATA}${LANG}${TEST}${DIMSUM}
./RNNMultiGRULoad.py ${ARGUMENTS}  > ${DATA}${LANG}${PREDICT}${TEST}${PRED}
echo "End train"

# add predict to the .dimsum --> predict_.dimsum
echo "Start add predict to "${DIMSUM}"."
echo ${DATA}${LANG}${TEST}${DIMSUM}" & "${DATA}${LANG}${PREDICT}${TEST}${PRED}" --> "${DATA}${LANG}${PREDICT}${TEST}${DIMSUM}
./addPredictToDimsumWithBIOVMWE.py ${OPT_DIMSUM} ${DATA}${LANG}${TEST}${DIMSUM} ${OPT_TAG} ${DATA}${LANG}${PREDICT}${TEST}${PRED} > ${DATA}${LANG}${PREDICT}${TEST}${DIMSUM}
echo "End add predict."

# .dimsum --> .cupt (predict_$test)
echo "Start parse "${DIMSUM}" to "${PREDICT}${CUPT}"."
echo ${DATA}${LANG}${PREDICT}${TEST}${DIMSUM}" --> "${RESULT}${LANG}${PREDICT}${TEST}${CUPT}
./dimsumWithGapsToCuptWithBIOVMWE.py ${OPT_DIMSUM}${DATA}${LANG}${PREDICT}${TEST}${DIMSUM} ${OPT_CUPT} ${DATA}${LANG}${TEST}${CUPT} > ${RESULT}${LANG}${PREDICT}${TEST_RESULT}${CUPT}
echo "End parse"${DATA}${LANG}${PREDICT}${TEST}${DIMSUM}" to "${RESULT}${LANG}${PREDICT}${TEST}${CUPT}"."
