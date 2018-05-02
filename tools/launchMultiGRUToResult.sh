#!/bin/bash

DATA="../data/sharedtask-data/1.1/"
RESULT="../result/BIO/"
LANG="BG/ DE/ EL/ EN/ ES/ EU/ FA/ FR/ HE/ HI/ HR/ HU/ IT/ LT/ PL/ PT/ RO/ SL/ TR/"
TRAIN="train"
DEV="dev"
CUPT=".cupt"
PRED=".pred"
DIMSUM=".dimsum"
PREDICT="predict-"
TEST="test.blind"
OPT_COLUMNS="--ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4"
OPT_TRAIN=" --train="
OPT_TEST=" --test="
OPT_TEST_BIS=" --test "
OPT_CUPT=" --cupt "
OPT_CUPT=" --cupt "
OPT_DIMSUM=" --dimsum "
OPT_TAG=" --tag "

# .cupt --> .dimsum  ${OPT_TEST_BIS}
for path in $LANG
do
echo "Start parse "${CUPT}" to "${DIMSUM}"."
echo ${DATA}${path}${TRAIN}${CUPT}
./parsemeCuptTrainToDimsum.py ${OPT_CUPT} ${DATA}${path}${TRAIN}${CUPT} > ${DATA}${path}${TRAIN}${DIMSUM}
echo ${DATA}${path}${DEV}${CUPT}
parsemeCuptTestToDimsum.py ${OPT_CUPT} ${DATA}${path}${TEST}${CUPT} > ${DATA}${path}${TEST}${DIMSUM}
echo "End parse "${CUPT}" to "${DIMSUM}"."
# train and predict
#./RNNMultiGRUWithVMWE.py --ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4 --train="$dimTrain" --test="$dimTest" > "$fileResult"
echo "train = "${DATA}${path}${TRAIN}${DIMSUM}" and test = "${DATA}${path}${TEST}${DIMSUM}
ARGUMENTS=${OPT_COLUMNS}${OPT_TRAIN}${DATA}${path}${TRAIN}${DIMSUM}${OPT_TEST}${DATA}${path}${TEST}${DIMSUM}
./RNNMultiGRU.py ${ARGUMENTS}  > ${DATA}${path}${PREDICT}${TEST}${PRED}
echo "End train"

# add predict to the .dimsum --> predict_.dimsum
echo "Start add predict to "${DIMSUM}"."
echo ${DATA}${path}${TEST}${DIMSUM}" & "${DATA}${path}${PREDICT}${TEST}${PRED}" --> "${DATA}${path}${PREDICT}${TEST}${DIMSUM}
./addPredictToDimsum.py ${OPT_DIMSUM} ${DATA}${path}${TEST}${DIMSUM} ${OPT_TAG} ${DATA}${path}${PREDICT}${TEST}${PRED} > ${DATA}${path}${PREDICT}${TEST}${DIMSUM}
echo "End add predict."

# .dimsum --> .cupt (predict_$test)
echo "Start parse "${DIMSUM}" to "${PREDICT}${CUPT}"."
echo ${DATA}${path}${PREDICT}${TEST}${DIMSUM}" --> "${DATA}${path}${PREDICT}${TEST}${CUPT}
./dimsumWithGapsToCupt.py ${OPT_DIMSUM}${DATA}${path}${PREDICT}${TEST}${DIMSUM} ${OPT_CUPT} ${DATA}${path}${TEST}${CUPT} > ${RESULT}${path}${PREDICT}${TEST}${CUPT}
echo "End parse"${DIMSUM}" to "${PREDICT}${CUPT}"."
done
