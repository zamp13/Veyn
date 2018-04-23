#!/bin/bash

DATA="../data/sharedtask-data/1.1/"
LANG="BG/ DE/ EL/ ES/ EU/ FA/ FR/ HE/ HR/ HU/ IT/ PL/ PT/ RO/ SL/ TR/"
LANGDEVEMPTY="EN/ HI/ LT/"
TRAIN="train"
DEV="dev"
CUPT=".cupt"
DIMSUM=".dimsum"
PREDICT="predict-"
OPT_COLUMNS="--ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4"
OPT_TRAIN=" --train="
OPT_TEST=" --test="
OPT_CUPT=" --cupt "

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


# train and predict
#./RNNMultiGRUWithVMWE.py --ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4 --train="$dimTrain" --test="$dimTest" > "$fileResult"
echo "Start train RNN wiht "${TRAIN}${DIMSUM}" and test RNN with "${DEV}${DIMSUM}"."
echo "Result in the file "${PREDICT}${DEV}${DIMSUM}" (format .dimsum)."
for path in $LANG
do
echo "train = "${DATA}${path}${TRAIN}${DIMSUM}" and test = "${DATA}${path}${DEV}${DIMSUM}
ARGUMENTS=${OPT_COLUMNS}${OPT_TRAIN}${DATA}${path}${TRAIN}${DIMSUM}${OPT_TEST}${DATA}${path}${DEV}${DIMSUM}
./RNNMultiGRU.py ${ARGUMENTS}  > ${DATA}${path}${PREDICT}${DEV}${DIMSUM}
done
echo "End train"

# add predict to the .dimsum --> predict_.dimsum
#echo "Start add predict to "${DIMSUM}"."
#for path in $LANG
#do
#echo ${DATA}${path}${DEV}${DIMSUM}" & "${DATA}${path}${PREDICT}${DEV}${DIMSUM}" --> "${DATA}${path}${PREDICT}${DEV}${DIMSUM}
#./addPredictToDimsum.py ${DATA}${path}${DEV}${DIMSUM} ${DATA}${path}${PREDICT}${DEV}${DIMSUM} > ${DATA}${path}${PREDICT}${DEV}${DIMSUM}
#done
#echo "End add predict."

# .dimsum --> .cupt (predict_$test)
#echo "Start parse "${DIMSUM}" to "${PREDICT}${CUPT}"."
#for path in $LANG
#do
#echo ${DATA}${path}${PREDICT}${DEV}${DIMSUM}" --> "${DATA}${path}${PREDICT}${DEV}${CUPT}
#./dimsumWithGapsToCupt.py ${DATA}${path}${PREDICT}${DEV}${DIMSUM} ${DATA}${path}${PREDICT}${DEV}${CUPT} > ${DATA}${path}${PREDICT}${DEV}${CUPT}
#done
#echo "End parse"${DIMSUM}" to "${PREDICT}${CUPT}"."
