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

# add predict to the .dimsum --> predict_.dimsum
echo "Start add predict to "${DIMSUM}"."
for path in $LANG
do
echo ${DATA}${path}${DEV}${DIMSUM}" & "${DATA}${path}${PREDICT}${DEV}${DIMSUM}" --> "${DATA}${path}${PREDICT}${DEV}${DIMSUM}
./addPredictToDimsum.py ${OPT_DIMSUM} ${DATA}${path}${DEV}${DIMSUM} ${OPT_TAG} ${DATA}${path}${PREDICT}${DEV}${PRED} > ${DATA}${path}${PREDICT}${DEV}${DIMSUM}
done
echo "End add predict."