#!/bin/bash

DATA="../data/sharedtask-data/1.1/"
LANG="BG/ DE/ EL/ ES/ EU/ FA/ FR/ HE/ HR/ HU/ IT/ PL/ PT/ RO/ SL/ TR/"
LANGDEVEMPTY="EN/ HI/ LT/"
TRAIN="train"
DEV="dev"
CUPT=".cupt"
PRED=".pred"
DIMSUM=".dimsum"
PREDICT="predict-"
OPT_COLUMNS="--ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4"
OPT_TRAIN=" --train="
OPT_TEST=" --test="
OPT_CUPT=" --cupt "
OPT_DIMSUM=" --dimsum "
OPT_TAG=" --tag "

# .dimsum --> .cupt (predict_$test)
#echo "Start parse "${DIMSUM}" to "${PREDICT}${CUPT}"."
for path in $LANG
do
echo ${DATA}${path}${PREDICT}${DEV}${DIMSUM}" --> "${DATA}${path}${PREDICT}${DEV}${CUPT}
./dimsumWithGapsToCupt.py ${OPT_DIMSUM} ${DATA}${path}${PREDICT}${DEV}${DIMSUM} ${OPT_CUPT} ${DATA}${path}${DEV}${CUPT} > ${DATA}${path}${PREDICT}${DEV}${CUPT}
done
echo "End parse"${DIMSUM}" to "${PREDICT}${CUPT}"."
