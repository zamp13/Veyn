#!/bin/bash

DATA="../data/sharedtask-data/1.1/"
RESULT="../result/BIOVMWE/"
LANG=${1}"/"
# BG/ DE/ EL/ EN/ ES/ EU/ FA/ FR/ HE/ HI/ HR/ HU/ IT/ LT/ PL/ PT/ RO/ SL/ TR/
MODELS="../Models/"${LANG}"model"${2}".h5"
TRAIN="train"
DEV_OR_TEST=${3}
CUPT=".cupt"
PRED=".pred"
DIMSUM=".dimsum"
PREDICT="predictBIOvmwe-"
OPT_COLUMNS="--ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4"
OPT_TRAIN=" --train="
OPT_TEST=" --test="
OPT_SAVE=" --save="
OPT_TEST_BIS=" --test "
OPT_CUPT=" --cupt "
OPT_CUPT=" --cupt "
OPT_DIMSUM=" --dimsum "
OPT_TAG=" --tag "

# .cupt --> .dimsum

echo "Start parse "${CUPT}" to "${DIMSUM}"."
echo ${DATA}${LANG}${TRAIN}${CUPT}
./parsemeCuptToDimsumWithBIOVMWE.py ${OPT_CUPT} ${DATA}${LANG}${TRAIN}${CUPT} > ${DATA}${LANG}${TRAIN}${DIMSUM}
echo ${DATA}${LANG}${DEV_OR_TEST}${CUPT}
./parsemeCuptToDimsumWithBIOVMWE.py ${OPT_CUPT} ${DATA}${LANG}${DEV_OR_TEST}${CUPT} ${OPT_TEST_BIS} > ${DATA}${LANG}${DEV_OR_TEST}${DIMSUM}
echo "End parse "${CUPT}" to "${DIMSUM}"."
# train and predict
#./RNNMultiGRUWithVMWE.py --ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4 --train="$dimTrain" --test="$dimTest" > "$fileResult"
echo "train = "${DATA}${LANG}${TRAIN}${DIMSUM}" and test = "${DATA}${LANG}${DEV_OR_TEST}${DIMSUM}
ARGUMENTS=${OPT_COLUMNS}${OPT_TRAIN}${DATA}${LANG}${TRAIN}${DIMSUM}${OPT_TEST}${DATA}${LANG}${DEV_OR_TEST}${DIMSUM}${OPT_SAVE}${MODELS}
./RNNMultiGRUWithBIOVMWE.py ${ARGUMENTS}  > ${DATA}${LANG}${PREDICT}${DEV_OR_TEST}${PRED}
echo "End train"

# add predict to the .dimsum --> predict_.dimsum
echo "Start add predict to "${DIMSUM}"."
echo ${DATA}${LANG}${DEV_OR_TEST}${DIMSUM}" & "${DATA}${LANG}${PREDICT}${DEV_OR_TEST}${PRED}" --> "${DATA}${LANG}${PREDICT}${DEV_OR_TEST}${DIMSUM}
./addPredictToDimsumWithBIOVMWE.py ${OPT_DIMSUM} ${DATA}${LANG}${DEV_OR_TEST}${DIMSUM} ${OPT_TAG} ${DATA}${LANG}${PREDICT}${DEV_OR_TEST}${PRED} > ${DATA}${LANG}${PREDICT}${DEV_OR_TEST}${DIMSUM}
echo "End add predict."

# .dimsum --> .cupt (predict_$test)
echo "Start parse "${DIMSUM}" to "${PREDICT}${CUPT}"."
echo ${DATA}${LANG}${PREDICT}${DEV_OR_TEST}${DIMSUM}" --> "${RESULT}${LANG}${PREDICT}${DEV_OR_TEST}${CUPT}
./dimsumWithGapsToCuptWithBIOVMWE.py ${OPT_DIMSUM}${DATA}${LANG}${PREDICT}${DEV_OR_TEST}${DIMSUM} ${OPT_CUPT} ${DATA}${LANG}${DEV_OR_TEST}${CUPT} > ${RESULT}${LANG}${PREDICT}${DEV_OR_TEST}${CUPT}
echo "End parse"${DATA}${LANG}${PREDICT}${DEV_OR_TEST}${DIMSUM}" to "${RESULT}${LANG}${PREDICT}${DEV_OR_TEST}${CUPT}"."

