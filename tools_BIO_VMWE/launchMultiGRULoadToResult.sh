#!/bin/bash

# Parameters:
# $1 -> Language code, uppercase, 2 letters
# $2 -> Nom du modèle - chaîne de caractères quelconque
# $3 -> Fichier pour faire les prédictions, sans extension.cupt : "dev" ou "test.blind"

DATA="../data/sharedtask-data/1.1/"
RESULT="../result/BIOVMWE/"
LANG=${1}
INPUTFOLDER="${DATA}${LANG}/"
RESULTFOLDER="${RESULT}${LANG}/"
# BG/ DE/ EL/ EN/ ES/ EU/ FA/ FR/ HE/ HI/ HR/ HU/ IT/ LT/ PL/ PT/ RO/ SL/ TR/
MODELS="../Models/BIOVMWE/"${LANG}/"model${2}.h5"
TRAIN="train"
DEV_OR_TEST=${3}
CUPT=".cupt"
PRED=".pred"
DIMSUM=".dimsum"
PREDICT="predictBIOvmwe-${2}-"
OPT_COLUMNS="--ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4"
OPT_TRAIN=" --train="
OPT_TEST=" --test="
OPT_SAVE=" --save="
OPT_TEST_BIS=" --test "
OPT_CUPT=" --cupt "
OPT_LOAD=" --load="
OPT_DIMSUM=" --dimsum "
OPT_TAG=" --tag "
OPT_PREINIT="  --embeddings ../data/embeddings/$LANG-col3-w2v-skipgram-size250-window5-mincount2-negative10.vectors,3:../data/embeddings/$LANG-col4-w2v-skipgram-size250-window5-mincount2-negative10.vectors,4"

# .cupt --> .dimsum
date
hostname

echo "Start parse "${CUPT}" to "${DIMSUM}"."
echo ${INPUTFOLDER}${TRAIN}${CUPT}
./parsemeCuptToDimsumWithBIOVMWE.py ${OPT_CUPT} ${INPUTFOLDER}${TRAIN}${CUPT} > ${INPUTFOLDER}${TRAIN}${DIMSUM}
echo ${INPUTFOLDER}${DEV_OR_TEST}${CUPT}
./parsemeCuptToDimsumWithBIOVMWE.py ${OPT_CUPT} ${INPUTFOLDER}${DEV_OR_TEST}${CUPT} ${OPT_TEST_BIS} > ${INPUTFOLDER}${DEV_OR_TEST}${DIMSUM}
echo "End parse "${CUPT}" to "${DIMSUM}"."
# train and predict
#./RNNMultiGRUWithVMWE.py --ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4 --train="$dimTrain" --test="$dimTest" > "$fileResult"
mkdir -p `dirname ${MODELS}` # Create folder for the model, if not existing
echo "train = "${INPUTFOLDER}${TRAIN}${DIMSUM}" and test = "${INPUTFOLDER}${DEV_OR_TEST}${DIMSUM}
ARGUMENTS="${OPT_COLUMNS}${OPT_TRAIN}${INPUTFOLDER}${TRAIN}${DIMSUM}${OPT_TEST}${INPUTFOLDER}${DEV_OR_TEST}${DIMSUM}${OPT_LOAD}${MODELS}${OPT_PREINIT}"
./RNNMultiGRUWithBIOVMWE.py ${ARGUMENTS}  > ${INPUTFOLDER}${PREDICT}${DEV_OR_TEST}${PRED}
echo "End train"

# add predict to the .dimsum --> predict_.dimsum
echo "Start add predict to "${DIMSUM}"."
echo ${INPUTFOLDER}${DEV_OR_TEST}${DIMSUM}" & "${INPUTFOLDER}${PREDICT}${DEV_OR_TEST}${PRED}" --> "${INPUTFOLDER}${PREDICT}${DEV_OR_TEST}${DIMSUM}
./addPredictToDimsumWithBIOVMWE.py ${OPT_DIMSUM} ${INPUTFOLDER}${DEV_OR_TEST}${DIMSUM} ${OPT_TAG} ${INPUTFOLDER}${PREDICT}${DEV_OR_TEST}${PRED} > ${INPUTFOLDER}${PREDICT}${DEV_OR_TEST}${DIMSUM}
echo "End add predict."

# .dimsum --> .cupt (predict_$test)
mkdir -p "${RESULTFOLDER}" # Create folder for the model, if not existing
echo "Start parse "${DIMSUM}" to "${PREDICT}${CUPT}"."
echo ${INPUTFOLDER}${PREDICT}${DEV_OR_TEST}${DIMSUM}" --> "${RESULTFOLDER}${PREDICT}${DEV_OR_TEST}${CUPT}
./dimsumWithGapsToCuptWithBIOVMWE.py ${OPT_DIMSUM}${INPUTFOLDER}${PREDICT}${DEV_OR_TEST}${DIMSUM} ${OPT_CUPT} ${INPUTFOLDER}${DEV_OR_TEST}${CUPT} > ${RESULTFOLDER}${PREDICT}${DEV_OR_TEST}${CUPT}
echo "End parse"${INPUTFOLDER}${PREDICT}${DEV_OR_TEST}${DIMSUM}" to "${RESULTFOLDER}${PREDICT}${DEV_OR_TEST}${CUPT}"."

date
hostname
