#!/bin/bash

DATA="../data/sharedtask-data/1.1/"
RESULT="../result/"
INDEX="1 2 3 4 5 6 7 8 9 10"
LANG=${1}"/"
RESULT="../result/BIO"
TRAIN="train"
DEV="dev"
CUPT=".cupt"
PRED=".pred"
DIMSUM=".dimsum"
TEXT=".txt"
PREDICT="predictvmwe-"
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
./parsemeCuptToDimsum.py ${OPT_CUPT} ${DATA}${LANG}${TRAIN}${CUPT} > ${DATA}${LANG}${TRAIN}${DIMSUM}
echo ${DATA}${LANG}${DEV}${CUPT}
./parsemeCuptToDimsum.py ${OPT_CUPT} ${DATA}${LANG}${DEV}${CUPT} > ${DATA}${LANG}${DEV}${DIMSUM}
echo "End parse "${CUPT}" to "${DIMSUM}"."

# train and predict
echo "Start train RNN."
ARGUMENTS=${OPT_COLUMNS}${OPT_TRAIN}${DATA}${LANG}${TRAIN}${DIMSUM}${OPT_TEST}${DATA}${LANG}${DEV}${DIMSUM}
for ind in $INDEX
do
echo "./RNNMultiGRUWithVMWE.py" ${ARGUMENTS} " > " ${RESULT}${LANG}${PREDICT}${DEV}${ind}${TEXT}
./RNNMultiGRU.py ${ARGUMENTS}  > ${RESULT}${LANG}${PREDICT}${DEV}${ind}${TEXT}
echo "End train "${ind}
done

python ../voteMajoritaire ${RESULT}${LANG} > ${RESULT}${LANG}${PREDICT}${DEV}${PRED}

./addPredictToDimsum.py ${OPT_DIMSUM} ${DATA}${LANG}${DEV}${DIMSUM} ${OPT_TAG} ${RESULT}${LANG}${PREDICT}${DEV}${PRED} > ${DATA}${LANG}${PREDICT}${DEV}${DIMSUM}

./dimsumWithGapsToCupt.py ${OPT_DIMSUM}${DATA}${LANG}${PREDICT}${DEV}${DIMSUM} ${OPT_CUPT} ${DATA}${LANG}${DEV}${CUPT} > ${DATA}${LANG}${PREDICT}${DEV}${CUPT}