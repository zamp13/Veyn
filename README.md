# Veyn

## Installation

It was developed with _python2.7_ and used _keras_ and _tensorflow_ free libraries.
Download all this libraries and clone the git to install its.

## Command

`./Veyn.py -h` to show all commands in a terminal.

#### Command table

| Commands | Required | Definition |
| :----: | :----: | :------------: |
| -h, --help | False | Helpers and print all commands in stdout |
| --ignoreColumns | True | To ignore some columns, and do not treat them as features.|
| --columnOfTags | True | To give the number of the column containing tags (default 4)|
|--embeddings| False | To give some files containing embeddings. First, you give the path of the file containing embeddings, and separate with a \",\" you gave the column concern by this file. eg: file1,2 file2,5|
| --file | True | Give a file in the Extended CoNLL-U (.cupt) format. You can only give one file to train/test a model. |
|--mode | True | If the file is a train file and you want to create a model.|
|--model | True | Name of the model which you want to save/load without extension.|
| --bio | False |   Option to use the representation of BIO. You can combine with other options like --gap or/and -mwe. You can't combine with --io option.|
| --io | False |   Option to use the representation of IO. You can combine with other options like --gap or/and -mwe. You can't combine with --bio option.|                    
| -g, --gap | False | Option to use the representation of BIO/IO with gap.|
|-mwe, --category|False |Option to use the representation of BIO/IO with VMWE.|
|--batch_size| True | Option to intialize the size of batch for the RNN (default 128).|
|--overlaps|False| Option to use the representation of BIO/IO with overlaps. We can't load a file test with overlaps. By default, if option test and overlaps are activated, only the option test is considered. |


