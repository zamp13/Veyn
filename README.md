# Veyn

## Installation

It was developed with _python2.7_ and used _keras_ and _tensorflow_ free libraries.
Download all this libraries and clone the git to install its.

We used directory `Model` to stock our models of system. Furthermore, this directory is so bigger and we can't push on the github directory.
But, you can create this repository with this command : `mkdir Model`.

### Data

We used the shared task corpora to create/tune this system. You can show data on the official website of the [PARSEME shared task 2018](multiword.sourceforge.net/sharedtask2018) or directly on there [gitlab repository](https://gitlab.com/parseme/sharedtask-data/tree/master).


## Command

`./bin/Veyn.py -h` to show all commands in a terminal.

#### Examples commands

Command to create and train a model:

`./bin/Veyn.py --file fileTest/trial-train.cupt --mode train --model Model/trial-model -cat`

Command to load and test a model:

`./bin/Veyn.py --file fileTest/trial-test.cupt --mode test --model Model/trial-model -cat`

#### Command table

| Commands | Required | Definition |
| :----: | :----: | :------------: |
| -h, --help | False | Helpers and print all commands in stdout |
| -feat, --featureColumns | False | To treat columns as features. The first column is number 1, the second 2... By default, features are LEMME and POS, e.g 3 4|
| --mweTags | False | To give the number of the column containing tags (default 11) Careful! The first column is number 1, the second number 2, ...|
|--embeddings| False | To give some files containing embeddings. First, you give the path of the file containing embeddings, and separate with a \",\" you gave the column concern by this file. eg: file1,2 file2,5 ... You could have only column match with featureColumns.|
| --file | True | Give a file in the Extended CoNLL-U (.cupt) format. You can only give one file to train/test a model. You can give a CoNLL file to only test it.|
|--mode | True | To choice the mode of the system : train/test. If the file is a train file and you want to create a model use \'train\'. If the file is a test/dev file and you want to load a model use \'test\'. In test mode the system doesn't need params RNN.|
|--model | True | Name of the model which you want to save/load without extension. e.g \'nameModel\' , and the system save/load files nameModel.h5, nameModel.json and nameModel.voc. nameModel.h5 is the model file. nameModel.voc is the vocabulary file. nameModel.args is the arguments file which train your model.|
| --io | False |   Option to use the representation of IO. You can combine with other options like --nogap or/and --cat. By default, the representation is BIO.|                    
| -ng, --ngap | False | Option to use the representation of BIO/IO without gap. By default, the gap it is using to the representation of BIO/IO.|
|-cat, --category|False |Option to use the representation of BIO/IO with categories. By default, the representation of BIO/IO is without categories.|
|--sentences_per_batch| False |Option to initialize the size of mini batch for the RNN. By default, batch_size is 128.|
|--max_sentence_size| False |Option to initialize the size of sentence for the RNN. By default, max_sentence_size is 200.|
|--overlaps|False| Option to use the representation of BIO/IO with overlaps. We can't load a file test with overlaps, if option test and overlaps are activated, only the option test is considered. By default, the representation is without overlaps. |
|--validation_split|False| Option to configure the validation_split to train the RNN. By default 0.3(30%) of train file is use to validation data.|
|--validation_data|False| Give a file in the Extended CoNLL-U (.cupt) format to loss function for the RNN.|
|--epochs|False| Number of epochs to train RNN. By default, RNN trains on 10 epochs.|
|--recurrent_unit|False| This option allows choosing the type of recurrent units in the recurrent layer. By default it is biGRU. You can choice GRU, LSTM, biGRU, biLSTM.|
|--number_recurrent_layer|False| This option allows choosing the numbers of recurrent layer. By default it is 2 recurrent layers.|
|--size_recurrent_layer|False| This option allows choosing the size of recurrent layer. By default it is 512.|
|--feat_embedding_size|False| Option that takes as input a sequence of integers corresponding to the dimension/size of the embeddings layer of each column given to the --feat option. By default, all embeddings have the same size, use the current default value (64)|
|--early_stopping_mode|False| Option to save the best model training in function of acc/loss value, only if you use validation_data or validation_split. By default, it is in function of the loss value.|
|--patience_early_stopping|False| Option to choice patience for the early stopping. By default, it is 5 epochs.|


