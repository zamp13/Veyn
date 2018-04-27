#!/bin/bash

DATA="../data/sharedtask-data/1.1/"
LANG=${1}"/"
TRAIN="train"
DEV="dev"
CUPT=".cupt"
PRED=".pred"
DIMSUM=".dimsum"
PREDICT="predictBIOvmwe-"
OPT_COLUMNS="--ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4"
OPT_TRAIN=" --train="
OPT_TEST=" --test="
OPT_CUPT=" --cupt "
OPT_CUPT=" --cupt "
OPT_DIMSUM=" --dimsum "
OPT_TAG=" --tag "

# .cupt --> .dimsum
echo "Start parse "${CUPT}" to "${DIMSUM}"."
echo ${DATA}${LANG}${TRAIN}${CUPT}
./parsemeCuptToDimsumWithBIOVMWE.py ${OPT_CUPT} ${DATA}${LANG}${TRAIN}${CUPT} > ${DATA}${LANG}${TRAIN}${DIMSUM}
echo ${DATA}${LANG}${DEV}${CUPT}
./parsemeCuptToDimsumWithBIOVMWE.py ${OPT_CUPT} ${DATA}${LANG}${DEV}${CUPT} > ${DATA}${LANG}${DEV}${DIMSUM}
echo "End parse "${CUPT}" to "${DIMSUM}"."

# train and predict
echo "Start train RNN."
ARGUMENTS=${OPT_COLUMNS}${OPT_TRAIN}${DATA}${LANG}${TRAIN}${DIMSUM}${OPT_TEST}${DATA}${LANG}${DEV}${DIMSUM}
echo "./RNNMultiGRUWithVMWE.py" ${ARGUMENTS} " > " ${DATA}${LANG}${PREDICT}${DEV}${DIMSUM}
./RNNMultiGRUWithBIOVMWE.py ${ARGUMENTS}  > ${DATA}${LANG}${PREDICT}${DEV}${PRED}
echo "End train"

# add predict to the .dimsum --> predict_.dimsum
echo "Start add predict to "${DIMSUM}"."
echo ${DATA}${LANG}${DEV}${DIMSUM}" & "${DATA}${LANG}${PREDICT}${DEV}${DIMSUM}" --> "${DATA}${LANG}${PREDICT}${DEV}${DIMSUM}
./addPredictToDimsumWithBIOVMWE.py ${OPT_DIMSUM} ${DATA}${LANG}${DEV}${DIMSUM} ${OPT_TAG} ${DATA}${LANG}${PREDICT}${DEV}${PRED} > ${DATA}${LANG}${PREDICT}${DEV}${DIMSUM}
echo "End add predict."

# .dimsum --> .cupt (predict_$test)
echo "Start parse "${DIMSUM}" to "${PREDICT}${CUPT}"."
echo ${DATA}${LANG}${PREDICT}${DEV}${DIMSUM}" --> "${DATA}${LANG}${PREDICT}${DEV}${CUPT}
./dimsumWithGapsToCuptWithBIOVMWE.py ${OPT_DIMSUM} ${DATA}${LANG}${PREDICT}${DEV}${DIMSUM} ${OPT_CUPT} ${DATA}${LANG}${DEV}${CUPT} > ${DATA}${LANG}${PREDICT}${DEV}${CUPT}
echo "End parse"${DIMSUM}" to "${PREDICT}${CUPT}"."
