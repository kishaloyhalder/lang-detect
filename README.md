# Neural Language Detection

This is a project to detect language from textual documents. The repository provides sources in python to achieve that. It has multiple functionalities. It can be used off-the-shelf with the pre-trained model. You can also train it on other datasets if you wish. On a dataset consisting of text samples from 21 different languages harvested from Wikipedia, the pre-trained model achieves a score of 94.55% on test dataset with unseen samples.

In the current implementation, the model uses a sub-word based embedding mechanism which is very useful for handling out-of-vocabulary words. After the tokenization, a Bidirectional LSTM is used to encode the sequence of tokens present in the input. Finally the prediction is made by a fully connected layer with `N` number of prediction heads, where `N` is the number of languages in the dataset.
The model is trained with back-propagation using `categorical cross-entropy` loss.

We observe a few interesting patterns in the model behaviour. Since it is trained based on sub-word based tokenization, for languages that use unique set of characters are very easy to distinguish for the model (as for example, `bengali`). However, for languages which have many characters common (such as `english` and `french`), the model sometimes gets confused and makes incorrect predictions.

## Files

* README.md : This file.
* main.py: The main python file to run. 
* train_model.py: This has all the modeling logic in it.
* extract_data.py: This can be used to populate new datasets from Wikipedia dump.
* utilities.py: An utility python file.
* data/extracted/21_train.tsv: Training file.
* data/extracted/21_val.tsv: Validation file.
* data/extracted/21_test.tsv: Test file.

## Supported languages by the pre-trained model
  The following languages (in ISO 639-1 codes) are supported.
```bg, bn, br, bs, ca, cs, cy, da, de, dz, el, en, eo, es, et, eu, fa, fi, fo, fr, ga```

## Dataset generation (optional):
  The datasets used for the pre-training is included in this dump. However if you wish to create a new dataset. Please follow the steps.
  1) Download the dump: Follow the instructions given here: https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2735
  2) Placement of data file: Place the downloaded `.gz` files in `./data` directory.
  3) Run the file `extract_data.py`. It would generate 3 files namely, `21_train.tsv`, `21_test.tsv`, `21_val.tsv` if 21 languages are selected in the beginning of the script. By default, the script selects first 20K lines (if present) from each language, and keeps 80% in train, and rest in dev, test (10% in each). The long documents are broken (delimited by space) into sequences of 20 tokens and then each is treated as a sample. These values can be changed by modifying the `extract_data.py` file, near the beginning.


## Instructions to run

* Required Data Files: 
  1) training/validation/test Files: The code expects to see these files depending on how
  you want to use them. All of them have the identical format. They have two columns i.e., `text` and `language`, separated by tab (`\t`). Each row in these files is a pair of text sample and the ISO 639-1 code for the source language of the sample, separated by tab.

  2) Model Files: The code expects model files if you want to use the pre-trained models. It is recommended to use the pre-trained model as much as possible, otherwise the sub-word encoding takes long to populate from scratch.
    * `model_weights.h5`: A serialized pre-trained keras model file.
    * `tokenizer_vocab.subwords`: A serialized tokenizer with 200K sub-word tokens.
    * `label_encoder.pkl`: A serialized one hot encoder trained with 21 languages.
  The model expects these files to be in the `./model` directory.
      
      
* Running the program:
  The program `main.py` provides the following user options to be specified during execution.
  1) `--train`: Path of training file if you want to train from scratch.
  2) `--val`: Path of validation file if you want to do early stopping.
  3) `--test`: Path of test file if you want to test.
  4) `--predict`: Path of a file where each line is a document to classify.
  5) `--sample`: A single sentence to classify.
  6) `--topwords`: Number of words in vocabulary.
  7) `--maxlen`: Max. length of input sequence in words.
  8) `--embedding`: Embedding dimension.
  9) `--size`: Model Dimension.
  10) `--epochs`: Number of epochs.
  11) `--batch`: Mini-batch size.
  12) `--randomseed`: Random seed to reproduce numbers.
  13) `--cached`: Use cached models for train.
  14) `--vocab`: Repopulate vocab.
  15) `--early`: Use early stopping for model training.
  Please refer to `main.py` for the default values.

  Example: 
  To train the model from scratch and evaluate it you may run:
      
      python3 main.py --train=data/extracted/21_train.tsv --val=data/extracted/21_val.tsv --test=data/extracted/21_test.tsv --early --vocab

  It would train the model on `train.tsv` data, and use `val.tsv` as validation data. Please have patience as the above command trains everything from scratch, and thus might take a long time to run.

  If you do not want to populate the vocabulary time and again, then just use the following

      python3 main.py --train=data/extracted/21_train.tsv --val=data/extracted/21_val.tsv --test=data/extracted/21_test.tsv --early

  Evaluation can be also done as a separate step with the pre-trained model.

      python3 main.py --test=data/extracted/21_test.tsv

  The above command will evaluate the model on the `test.tsv` data. It reports the overall prediction accuracy and the confusion matrix in the log file generated in the `./results` directory.

  If you would just like to predict the languages for a bunch of text sentences, you may run:
      
      python3 main.py --predict=data/test.tsv

  In this case, the predictions would be output in the log file in the `./results` directory.

  The code provides a super simple way to make prediction for a single sample as well. Just run:

      python3 main.py --sample="I like to play electric guitar"

  The above command would dump the result on the console. You are welcome :-)

    
## Requirements
 
 The following packages are required to run the code
 
 * Python 3
 * Tensorflow
 * Keras
 * Sklearn
 * pandas
