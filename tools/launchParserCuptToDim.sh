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

# .cupt --> .dimsum
echo "Start parse "${CUPT}" to "${DIMSUM}"."
for path in $LANG
do
echo ${DATA}${path}${TRAIN}${CUPT}
./parsemeCuptToDimsum.py ${OPT_CUPT} ${DATA}${path}${TRAIN}${CUPT} > ${DATA}${path}${TRAIN}${DIMSUM}
echo ${DATA}${path}${DEV}${CUPT}
./parsemeCuptToDimsum.py ${OPT_CUPT} ${DATA}${path}${DEV}${CUPT} > ${DATA}${path}${DEV}${DIMSUM}
done

for path in ${LANGDEVEMPTY}
do
echo ${DATA}${path}${TRAIN}${CUPT}
./parsemeCuptToDimsum.py ${OPT_CUPT} ${DATA}${path}${TRAIN}${CUPT} > ${DATA}${path}${TRAIN}${DIMSUM}
done
echo "End parse "${CUPT}" to "${DIMSUM}"."
