#!/bin/bash

DATA="../data/sharedtask-data/1.1/"
LANG=${1}"/"
TRAIN="train"
DEV="dev"
CUPT=".cupt"
DIMSUM=".dimsum"
PREDICT="predict-"
OPT_COLUMNS="--ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4"
OPT_TRAIN=" --train="
OPT_TEST=" --test="


# .cupt --> .dimsum
echo "Start parse "${CUPT}" to "${DIMSUM}"."
echo ${DATA}${LANG}${TRAIN}${CUPT}
./parsemeFileToDimsum.py ${DATA}${LANG}${TRAIN}${CUPT} > ${DATA}${LANG}${TRAIN}${DIMSUM}
echo ${DATA}${LANG}${DEV}${CUPT}
./parsemeFileToDimsum.py ${DATA}${LANG}${DEV}${CUPT} > ${DATA}${LANG}${DEV}${DIMSUM}
echo "End parse "${CUPT}" to "${DIMSUM}"."

# train and predict
echo "Start train RNN."
ARGUMENTS=${OPT_COLUMNS}${OPT_TRAIN}${DATA}${LANG}${TRAIN}${DIMSUM}${OPT_TEST}${DATA}${LANG}${DEV}${DIMSUM}
echo "./RNNMultiGRU.py" ${ARGUMENTS} " > " ${DATA}${LANG}${PREDICT}${DEV}${DIMSUM}
./RNNMultiGRU.py ${ARGUMENTS}  > ${DATA}${LANG}${PREDICT}${DEV}${DIMSUM}
echo "End train"

# add predict to the .dimsum --> predict_.dimsum
echo "Start add predict to "${DIMSUM}"."
echo ${DATA}${LANG}${DEV}${DIMSUM}" & "${DATA}${LANG}${PREDICT}${DEV}${DIMSUM}" --> "${DATA}${LANG}${PREDICT}${DEV}${DIMSUM}
./addPredictToDimsum.py ${DATA}${LANG}${DEV}${DIMSUM} ${DATA}${LANG}${PREDICT}${DEV}${DIMSUM} > ${DATA}${LANG}${PREDICT}${DEV}${DIMSUM}
echo "End add predict."

# .dimsum --> .cupt (predict_$test)
echo "Start parse "${DIMSUM}" to "${PREDICT}${CUPT}"."
echo ${DATA}${LANG}${PREDICT}${DEV}${DIMSUM}" --> "${DATA}${LANG}${PREDICT}${DEV}${CUPT}
./dimsumWithGapsToCupt.py ${DATA}${LANG}${PREDICT}${DEV}${DIMSUM} ${DATA}${LANG}${PREDICT}${DEV}${CUPT} > ${DATA}${LANG}${PREDICT}${DEV}${CUPT}
echo "End parse"${DIMSUM}" to "${PREDICT}${CUPT}"."
