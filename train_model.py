"""Implementation of a deep learning based model to detect language from 
   textual documents.
"""
import os
import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, Lambda
from keras.layers.wrappers import Bidirectional
from keras.layers.embeddings import Embedding
import keras.preprocessing as preprocessing
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

import tensorflow_datasets as tfds

from sklearn.preprocessing import OneHotEncoder
#from sklearn.externals import joblib
import joblib
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import pandas as pd

from utilities import Logger
from datetime import datetime

TOKENIZER_PATH = 'model/tokenizer_vocab'
LABEL_ENCODER_PATH = 'model/label_encoder.pkl'
MODEL_PATH = 'model/model_weights.h5'
LOG_FILE_PATH = 'results/'+datetime.now().strftime('%H_%M_%d_%m_%Y.log')

logger = Logger(LOG_FILE_PATH)

def prepare_sequences(tokenizer, texts, options):
	"""Tokenizes the textual input and prepares sequences by applying padding"""
	text = [tokenizer.encode(sample) for sample in texts]
	text = preprocessing.sequence.pad_sequences(text, maxlen=options.maxlen)
	return text

def train(options):
	"""Loads data, trains the language detection model."""
	train_data = pd.read_csv(options.train, sep='\t')
	train_examples = train_data.text.values.astype(str)

	train_labels = train_data.language.values.astype(str)
	
	if options.cached:
		tokenizer, label_encoder, model = load_cached_models()
	else:
		if options.vocab:
			#create tokenizer from scratch
			tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((text for \
				text in train_examples), target_vocab_size=options.topwords)
			tokenizer.save_to_file(TOKENIZER_PATH)
			logger.log('tokenizer saved to: '+TOKENIZER_PATH)

			#create label encoder from scratch
			label_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
			label_encoder.fit(train_labels.reshape(-1, 1))
			joblib.dump(label_encoder, LABEL_ENCODER_PATH)
			logger.log('label encoder saved to: '+LABEL_ENCODER_PATH)

		else:
			tokenizer, label_encoder, _ = load_cached_models()

		num_classes = len(label_encoder.categories_[0])

		# create the model from scratch
		model = Sequential()
		model.add(Embedding(options.topwords, options.embedding, mask_zero=True))
		model.add(Bidirectional(LSTM(options.size, dropout=0.2, recurrent_dropout=0.2)))
		model.add(Dense(num_classes, activation='softmax'))
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		#print(model.summary())
		stringlist = []
		model.summary(print_fn=lambda x: stringlist.append(x))
		logger.log("\n".join(stringlist))


	X_train = prepare_sequences(tokenizer, train_examples, options)
	y_train = label_encoder.transform(train_labels.reshape(-1, 1))
	

	if options.early:
		val_data = pd.read_csv(options.val, sep='\t')

		val_examples = val_data.text.values.astype(str)
		val_labels = val_data.language.values.astype(str)

		X_val = prepare_sequences(tokenizer, val_examples, options)
		y_val = label_encoder.transform(val_labels.reshape(-1, 1))

		es = EarlyStopping(monitor='val_accuracy', mode='max', patience=2)
		mc = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
		history = model.fit(X_train, y_train, epochs=options.epochs, batch_size=options.batch, shuffle=True, validation_data=(X_val, y_val), callbacks=[es, mc])	
	else:
		history = model.fit(X_train, y_train, epochs=options.epochs, batch_size=options.batch, shuffle=True)
		model.save(MODEL_PATH)
	
	logger.log('main model weights saved to: '+MODEL_PATH)
	logger.log('training history:')
	logger.log(process_history(history))

def test(options):
	"""Evaluates the model based on accuracy. Also outputs a confusion matrix."""
	test_data = pd.read_csv(options.test, sep='\t')

	test_examples = test_data.text.values.astype(str)
	test_labels = test_data.language.values.astype(str)

	tokenizer, label_encoder, model = load_cached_models()

	X_test = prepare_sequences(tokenizer, test_examples, options)
	y_test = label_encoder.transform(test_labels.reshape(-1, 1))


	# Final evaluation of the model
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))
	logger.log("Accuracy: %.2f%%" % (scores[1]*100))

	predictions = model.predict(X_test, verbose=0)
	predicted_labels = label_encoder.inverse_transform(predictions)

	c_matrix = confusion_matrix(test_labels, predicted_labels, labels=label_encoder.categories_[0])
	logger.log(serialize_c_matrix(c_matrix, label_encoder.categories_[0]))

def predict(options):
	"""Predicts the language for some inputs. Outputs them in the log file."""
	test_data = pd.read_csv(options.predict, sep='\t')

	test_examples = test_data.text.values.astype(str)

	tokenizer, label_encoder, model = load_cached_models()

	X_test = prepare_sequences(tokenizer, test_examples, options)

	predictions = model.predict(X_test, verbose=0)

	predicted_langs = label_encoder.inverse_transform(predictions)
	logger.log('Predictions for input file: '+options.predict)
	logger.log(combine_x_p(test_data, predicted_langs))
	print('Predictions are written to: '+LOG_FILE_PATH)

def predict_individual(options):
	"""Predicts the language for a particular sample. Outputs on the console."""
	tokenizer, label_encoder, model = load_cached_models()
	X_test = prepare_sequences(tokenizer, [options.sample.encode()], options)

	prediction = model.predict(X_test, verbose=0)

	predicted_lang = label_encoder.inverse_transform(prediction)
	print('sample: '+options.sample)
	print('predicted language: '+str(predicted_lang[0]))

	return predicted_lang

def serialize_c_matrix(matrix, categories):
	"""Serializes a confusion matrix to a printable string."""
	st = '\t'
	for category in categories:
		st += str(category)+'\t'
	st += '\n'

	for i in range(0, len(categories)):
		st += str(categories[i]) +'\t'
		for j in range(0, len(categories)):
			st += str(matrix[i][j]) +'\t'
		st += '\n'
	return st	

def combine_x_p(data, predictions):
	data['predicted_language'] = predictions
	st = data.to_string(index=False)
	return st

def process_history(history):
	"""Serializes training history to string."""
	st = ""
	keys = history.history.keys()
	for key in keys:
		st += key + '\n'
		for value in history.history[key]:
			st += str(value)+'\n'
		st += '\n'
	return st

def load_cached_models():
	"""Loads pretrained models from the disk."""
	#tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(TOKENIZER_PATH)
	tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(TOKENIZER_PATH)
	logger.log('tokenizer loaded from: '+TOKENIZER_PATH)

	label_encoder = joblib.load(LABEL_ENCODER_PATH)
	logger.log('label encoder loaded from: '+LABEL_ENCODER_PATH)

	model = load_model(MODEL_PATH)
	logger.log('model weights loaded from: '+MODEL_PATH)
	
	return tokenizer, label_encoder, model


def run(options):
	"""Handles the logic based on user-options."""
	numpy.random.seed(options.randomseed)
	
	print('Outputs are being logged at: '+LOG_FILE_PATH)
	logger.log('arguments:'+str(vars(options)))

	if options.train is not None:
		train(options)

	if options.test is not None:
		test(options)

	if options.predict is not None:
		predict(options)

	if options.sample is not None:
		predict_individual(options)

	logger.log('finished')