#!/bin/bash

DATA="../data/sharedtask-data/1.1/"
RESULT="../result/BIOVMWE/"
# BG/ DE/ EL/ EN/ ES/ EU/ FA/ FR/ HE/ HI/ HR/ HU/ IT/ LT/ PL/ PT/ RO/ SL/ TR/
LANG=${1}"/"
MODELS="../Models/"${LANG}"model"${2}".h5"
TRAIN="train"
CUPT=".cupt"
PRED=".pred"
DIMSUM=".dimsum"
TEST_OR_DEV="test.blind"
TEST_RESULT="test.system"
PREDICT="predictBIOvmwe-"
OPT_COLUMNS="--ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4"
OPT_LOAD=" --load="
OPT_TEST=" --test="
OPT_TEST_BIS=" --test "
OPT_CUPT=" --cupt "
OPT_CUPT=" --cupt "
OPT_DIMSUM=" --dimsum "
OPT_TAG=" --tag "

# .cupt --> .dimsum
echo "Start parse "${CUPT}" to "${DIMSUM}"."
echo ${DATA}${LANG}${TEST_OR_DEV}${CUPT}
./parsemeCuptToDimsumWithBIOVMWE.py ${OPT_CUPT} ${DATA}${LANG}${TEST_OR_DEV}${CUPT} ${OPT_TEST_BIS} > ${DATA}${LANG}${TEST_OR_DEV}${DIMSUM}
echo "End parse "${CUPT}" to "${DIMSUM}"."
# train and predict
#./RNNMultiGRUWithVMWE.py --ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4 --train="$dimTrain" --test="$dimTest" > "$fileResult"
echo "train = "${DATA}${LANG}${TRAIN}${DIMSUM}" and test = "${DATA}${LANG}${TEST_OR_DEV}${DIMSUM}
ARGUMENTS=${OPT_COLUMNS}${OPT_LOAD}${MODELS}${OPT_TEST}${DATA}${LANG}${TEST_OR_DEV}${DIMSUM}
./RNNMultiGRULoad.py ${ARGUMENTS}  > ${DATA}${LANG}${PREDICT}${TEST_OR_DEV}${PRED}
echo "End train"

# add predict to the .dimsum --> predict_.dimsum
echo "Start add predict to "${DIMSUM}"."
echo ${DATA}${LANG}${TEST_OR_DEV}${DIMSUM}" & "${DATA}${LANG}${PREDICT}${TEST_OR_DEV}${PRED}" --> "${DATA}${LANG}${PREDICT}${TEST_OR_DEV}${DIMSUM}
./addPredictToDimsumWithBIOVMWE.py ${OPT_DIMSUM} ${DATA}${LANG}${TEST_OR_DEV}${DIMSUM} ${OPT_TAG} ${DATA}${LANG}${PREDICT}${TEST_OR_DEV}${PRED} > ${DATA}${LANG}${PREDICT}${TEST_OR_DEV}${DIMSUM}
echo "End add predict."

# .dimsum --> .cupt (predict_$test)
echo "Start parse "${DIMSUM}" to "${PREDICT}${CUPT}"."
echo ${DATA}${LANG}${PREDICT}${TEST_OR_DEV}${DIMSUM}" --> "${RESULT}${LANG}${PREDICT}${TEST_OR_DEV}${CUPT}
./dimsumWithGapsToCuptWithBIOVMWE.py ${OPT_DIMSUM}${DATA}${LANG}${PREDICT}${TEST_OR_DEV}${DIMSUM} ${OPT_CUPT} ${DATA}${LANG}${TEST_OR_DEV}${CUPT} > ${RESULT}${LANG}${PREDICT}${TEST_RESULT}${CUPT}
echo "End parse"${DATA}${LANG}${PREDICT}${TEST_OR_DEV}${DIMSUM}" to "${RESULT}${LANG}${PREDICT}${TEST_OR_DEV}${CUPT}"."
