#!/bin/bash

DATA="../data/sharedtask-data/1.1/"
RESULT="../result/BIOVMWE/"
LANG=${1}"/"
# BG/ DE/ EL/ EN/ ES/ EU/ FA/ FR/ HE/ HI/ HR/ HU/ IT/ LT/ PL/ PT/ RO/ SL/ TR/
TRAIN="train"
DEV="dev"
CUPT=".cupt"
PRED=".pred"
DIMSUM=".dimsum"
TEST="test.blind"
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
for path in $LANG
do
echo "Start parse "${CUPT}" to "${DIMSUM}"."
echo ${DATA}${path}${TRAIN}${CUPT}
./parsemeCuptToDimsumWithBIOVMWE.py ${OPT_CUPT} ${DATA}${path}${TRAIN}${CUPT} > ${DATA}${path}${TRAIN}${DIMSUM}
echo ${DATA}${path}${TEST}${CUPT}
./parsemeCuptToDimsumWithBIOVMWE.py ${OPT_CUPT} ${DATA}${path}${DEV}${CUPT} ${OPT_TEST_BIS} > ${DATA}${path}${DEV}${DIMSUM}
echo "End parse "${CUPT}" to "${DIMSUM}"."
# train and predict
#./RNNMultiGRUWithVMWE.py --ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4 --train="$dimTrain" --test="$dimTest" > "$fileResult"
echo "train = "${DATA}${path}${TRAIN}${DIMSUM}" and test = "${DATA}${path}${TEST}${DIMSUM}
ARGUMENTS=${OPT_COLUMNS}${OPT_TRAIN}${DATA}${path}${TRAIN}${DIMSUM}${OPT_TEST}${DATA}${path}${DEV}${DIMSUM}
./RNNMultiGRUWithBIOVMWE.py ${ARGUMENTS}  > ${DATA}${path}${PREDICT}${DEV}${PRED}
echo "End train"

# add predict to the .dimsum --> predict_.dimsum
echo "Start add predict to "${DIMSUM}"."
echo ${DATA}${path}${TEST}${DIMSUM}" & "${DATA}${path}${PREDICT}${TEST}${PRED}" --> "${DATA}${path}${PREDICT}${TEST}${DIMSUM}
./addPredictToDimsumWithBIOVMWE.py ${OPT_DIMSUM} ${DATA}${path}${DEV}${DIMSUM} ${OPT_TAG} ${DATA}${path}${PREDICT}${DEV}${PRED} > ${DATA}${path}${PREDICT}${DEV}${DIMSUM}
echo "End add predict."

# .dimsum --> .cupt (predict_$test)
echo "Start parse "${DIMSUM}" to "${PREDICT}${CUPT}"."
echo ${DATA}${path}${PREDICT}${TEST}${DIMSUM}" --> "${RESULT}${path}${PREDICT}${TEST}${CUPT}
./dimsumWithGapsToCuptWithBIOVMWE.py ${OPT_DIMSUM}${DATA}${path}${PREDICT}${DEV}${DIMSUM} ${OPT_CUPT} ${DATA}${path}${DEV}${CUPT} > ${RESULT}${path}${PREDICT}${DEV}${CUPT}
echo "End parse"${DATA}${path}${PREDICT}${TEST}${DIMSUM}" to "${RESULT}${path}${PREDICT}${TEST}${CUPT}"."
done
