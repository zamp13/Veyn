#!/bin/bash

#./find_parameters.py --ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4 --train=../train.dimsum --test=../test.dimsum --embeddings=../../data/embeddings/data.profiles,2 > hyperparamtest.txt

#./RNNTestDifferentParam.py --ignoreColumns=4:0:1 --columnOfTags=4 --train=../train.conll --test=../test.conll --embeddings=../../data/embeddings/data.profiles,2 > ../results/parseme/parsemeTags/tags/testNewTagsDifferentParam1.txt


./RNNMultiGRU.py --ignoreColumns=4:6:7:8:5:0:1 --columnOfTags=4 --train=../ftbTrain.dimsum --test=../ftbTest.dimsum --embeddings=../../data/embeddings/data.profiles,2 > ../results/parseme/BIO/FR/tags/timeLSTMEmbed.txt
