"""Driver code for language detection model."""
import train_model
import argparse
import sys 

def parse_user_options(args):
	parser = argparse.ArgumentParser(description="Parses command.")
	parser.add_argument("-t", "--train", help="Path of training file if you want to train from scratch", required=False)
	parser.add_argument("-d", "--val", help="Path of validation file if you want to do early stopping", required=False)
	parser.add_argument("-v", "--test", help="Path of test file if you want to test", required=False)
	parser.add_argument("-p", "--predict", help="Path of a file where each line is a document to classify", required=False)
	parser.add_argument("-m", "--sample", help="A single sentence to classify", required=False)
	parser.add_argument("-w", "--topwords", help="Number of words in vocab.", type=int, default=200000)
	parser.add_argument("-l", "--maxlen", help="Max len of input sequence in words.", type=int, default=20)
	parser.add_argument("-e", "--embedding", help="Embedding dimesion.", type=int, default=8)
	parser.add_argument("-s", "--size", help="Model Dimension.", type=int, default=100)
	parser.add_argument("-o", "--epochs", help="Number of epochs.", type=int, default=30)
	parser.add_argument("-b", "--batch", help="Mini-batch size.", type=int, default=32)
	parser.add_argument("-r", "--randomseed", help="Random seed to reproduce numbers.", type=int, default=7)
	parser.add_argument("-c", "--cached", help="Use cached models for train, test.", action="store_true")
	parser.add_argument("-u", "--vocab", help="Repopulate vocab.", action="store_true")
	parser.add_argument("-a", "--early", help="Use early stopping for model training.", action="store_true")
	options = parser.parse_args(args)
	return options

if __name__=='__main__':
	options = parse_user_options(sys.argv[1:])
	train_model.run(options)


