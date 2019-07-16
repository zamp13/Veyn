#! /usr/bin/env python
# -*- coding:UTF-8 -*-

################################################################################
#
# Copyright 2010-2014 Carlos Ramisch, Vitor De Araujo, Silvio Ricardo Cordeiro,
# Sandra Castellanos
#
# candidates.py is part of mwetoolkit
#
# mwetoolkit is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# mwetoolkit is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with mwetoolkit.  If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

import numpy as np
from gensim.models import Word2Vec
import sys


class PreprocessingW2V:
    r"""
        @class PreprocessingW2V
        This class is to use word2vec representation.
    """

    def __init__(self, model_name, train=True, size=128, min_count=1, window=5, epochs=10):
        self.model_name = model_name
        self.size = size
        self.new_train = train
        self.epochs = epochs
        if train:
            self.model = Word2Vec(min_count=min_count, size=self.size, window=window)
        else:
            try:
                self.model = Word2Vec.load(self.model_name)
            except Exception:
                print("Error model w2v", file=sys.stderr)

    def train(self, sentences):
        r"""
            To train Word2Vec model.
        :param sentences:
        :param epochs:
        :return:
        """
        print("Word2Vec model is training", file=sys.stderr)
        self.model.build_vocab(sentences)
        self.model.train(sentences, epochs=self.epochs, total_examples=self.model.corpus_count)
        self.model.save(self.model_name)
        print("End training and saving Word2Vec model", file=sys.stderr)

    def save_embeddings_w2v_format(self, name_embeddings):
        r"""
            Save the embeddings representation at w2v format to use embeddings trained on other model.
        :param name_embeddings:
        :return:
        """
        self.model.wv.save_word2vec_format(name_embeddings)

    def similarity_unk_vocab(self, vocab, sentences, index_col):
        r"""
            To change a unknown word to an other similar in vocab for all sentences in test.
        :param vocab:
        :param sentences:
        :return:
        """
        for sentence in sentences:
            for index_line in range(len(sentence)):
                line = sentence[index_line]
                if line != "\n":
                    line_TMP = line.rsplit("\t")
                    if line_TMP[index_col] not in vocab:
                        try:
                            similar_word = self.model.wv.most_similar(positive=line_TMP[index_col], topn=10)
                            flag_similar = True

                            for word in similar_word:
                                if word[0] in vocab and flag_similar:
                                    # print("Word : ", line_TMP[index_col], "is similar to ", word[0],
                                    #      file=sys.stderr)
                                    flag_similar = False
                                    line_TMP[index_col] = word[0]
                                    new_line = ""
                                    for index_columns in range(len(line_TMP)):
                                        new_line += line_TMP[index_columns] + "\t"
                                    sentence[index_line] = new_line
                        except Exception as exc:
                            print(exc, file=sys.stderr)
                            continue

    def matrix_embeddings(self, vocab):
        r"""
            Create a matrix embeddings of vocab.
        :param vocab: vocabulary of features
        :return: np.array((len(vocab), self.size), dtype=np.float32)
        """
        embeddings = np.zeros((len(vocab), self.size), dtype=np.float32)
        if '0' in self.model.wv:
            embeddings[vocab["<unk>"]] = np.array(self.model.wv['0'])
        for word in vocab:
            if word != "<unk>" and word != "0":
                try:
                    embeddings[vocab[word]] = self.model.wv[word]
                except Exception as e:
                    print(e, file=sys.stderr)
        return embeddings
