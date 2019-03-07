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
from gensim.models import FastText
import sys


def create_vocab(sentences):
    r"""
        Create vocabulary of sentences
    """
    vocab = {}
    vocab["<unk>"] = len(vocab)
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def add_tri_gramm_to_vocab(vocab):
    keys = list(vocab.keys())
    for key in keys:
        if key != "<unk>":
            if len(key) > 3:
                for index_c in range(len(key) - 2):
                    tri_gram = key[index_c] + key[index_c + 1] + key[index_c + 2]
                    if tri_gram not in vocab:
                        vocab[tri_gram] = len(vocab)


def write_embedings(vocab, fasttext, filename):
    file = open(filename, encoding="utf-8", mode="w")
    line = str(len(vocab)) + " " + str(len(fasttext["0"]))
    file.writelines(line + "\n")
    print("Start writing embeddings", file=sys.stderr)
    vector = [str(x) for x in fasttext["0"]]
    line = "<unk>"
    for x in vector:
        line += " " + x
    file.writelines(line + "\n")

    for word in vocab:
        if word != "<unk>":
            vector = [str(x) for x in fasttext.word_vec(word=word)]
            line = word
            for x in vector:
                line += " " + x
            file.writelines(line + "\n")
    file.close()
    print("End writing embeddings", file=sys.stderr)


class PreprocessingFasttext():
    r"""
        @class PreprocessingFasttext
        This class is to use fasttext representation.
    """

    def __init__(self, model_name, train=True, size=128, min_count=1, word_ngram=3, window=5, epochs=10):
        self.model_name = model_name
        self.size = size
        self.new_train = train
        self.epochs = epochs
        if train:
            self.model = FastText(min_count=min_count, size=self.size, word_ngrams=word_ngram, window=window)
        else:
            self.model = FastText.load(self.model_name)

    def train(self, sentences):
        r"""
            To train Fasttext model.
        :param sentences:
        :param epochs:
        :return:
        """
        print("Fasttext model is training", file=sys.stderr)
        self.model.build_vocab(sentences)
        self.model.train(sentences, epochs=self.epochs, total_examples=self.model.corpus_count)
        self.model.save(self.model_name)
        print("End training and saving Fasttext model", file=sys.stderr)

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
                            print("Word : ", line_TMP[index_col], "is similar to ", similar_word, file=sys.stderr)
                            for word in similar_word:
                                if word[0] in vocab and flag_similar:
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
        embeddings[vocab["<unk>"]] = np.array(self.model['<unk>'])
        for word in vocab:
            if word != "<unk>" and word != "0":
                embeddings[vocab[word]] = self.model[word]
        return embeddings


"""
if __name__ == "__main__":
    file = open("../sharedtask-data-master/1.1/FR/train.cupt", encoding="utf-8", mode="r")
    column_tags = 10
    column_sentence = 2
    filecupt = reader.ReaderCupt("BIOgcat", False, False, file, column_tags)
    filecupt.run()
    print("Start training embeddings", file=sys.stderr)
    sentences = filecupt.construct_sentence(column_sentence)
    vocab = create_vocab(sentences)
    add_tri_gramm_to_vocab(vocab)
    model = FastText(min_count=1, size=128, word_ngrams=3)
    model.build_vocab(sentences)
    model.train(sentences, epochs=10, total_examples=model.corpus_count)
    print("End training embeddings", file=sys.stderr)
    # write_embedings(vocab, model.wv, "../embeddings/fasttext-test-n_gram.embed")
    test = model.wv.similar_by_word("prendre")
    # model.wv.save_word2vec_format("../fasttext-test-fr-lemme.embed")
    # model.save("../fasttext-test-fr")
"""
