#/bin/python3

import os
import numpy as np
from botmodel import CHATBOT_MODEL
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class BOT:
	def __init__(self,num_samples=4000,
			max_num_words=10000,
			max_len_input=40,
			max_len_target=40,
			latent_dim=256,):
		self.NUM_SAMPLES = num_samples
		self.MAX_NUM_WORDS = max_num_words
		self.MAX_LEN_INPUT = max_len_input
		self.MAX_LEN_TARGET = max_len_target
		self.LATENT_DIM = latent_dim
		self.tokenizer = None

	def load_dataset(self,filename,reverse=False):
		t = 0
		input_texts = []
		target_texts = []
		target_input_texts = []
		for line in open(filename):
			t += 1
			if t > self.NUM_SAMPLES:
				break
			if '\t' not in line:
				continue
			if not reverse:
				input_text, target, *rest = line.rstrip().split('\t')
			else:
				target, input_text, *rest = line.rstrip().split('\t')

			target_text = target + ' <eos>'
			target_input_text = '<sos> ' + target
			if len(input_text.split(" ")) < 40 and len(target_text.split(" ")) < 40:
				input_texts.append(input_text)
				target_texts.append(target_text)
				target_input_texts.append(target_input_text)

		print ("Dataset size is of:%s" %len(input_texts))
		self.input_texts = input_texts
		self.target_texts = target_texts
		self.target_input_texts = target_input_texts

	def get_tokenizer(self,vocabfile):
		self.vocab = []
		for f in open(vocabfile):
			self.vocab.append(f.rstrip())
		tokenizer = Tokenizer(num_words=self.MAX_NUM_WORDS,filters='')
		tokenizer.fit_on_texts(self.vocab)
		self.word2idx = tokenizer.word_index
		print ("Found total unique keys for input %d" %len(self.word2idx))
		self.tokenizer = tokenizer

	def preprocess_data(self,vocabfile):
		if self.tokenizer is None:
			self.get_tokenizer(vocabfile)
		input_sequences = self.tokenizer.texts_to_sequences(self.input_texts)

		target_sequences = self.tokenizer.texts_to_sequences(self.target_texts)
		target_sequences_inputs = self.tokenizer.texts_to_sequences(self.target_input_texts)

		self.encoder_inputs = pad_sequences(input_sequences,maxlen=self.MAX_LEN_INPUT)

		self.decoder_inputs = pad_sequences(target_sequences_inputs,maxlen=self.MAX_LEN_TARGET,padding='post')
		self.decoder_targets = pad_sequences(target_sequences,maxlen=self.MAX_LEN_TARGET,padding='post')

	def process_pretrained(self,embedding_dim=100):
		self.EMBEDDING_DIM = embedding_dim
		word2vec = {}
		with open(os.path.join('../large_files/glove.6B/glove.6B.%sd.txt'%self.EMBEDDING_DIM)) as f:
			for line in f:
				values = line.split()
				word = values[0]
				vec = np.asarray(values[1:],dtype='float32')
				word2vec[word] = vec

		print ('Found %s word vectors'%len(word2vec))

		self.NUM_WORDS = min(self.MAX_NUM_WORDS, len(self.word2idx) + 1)
		self.embedding_matrix = np.zeros((self.NUM_WORDS, self.EMBEDDING_DIM))
		for word,i in self.word2idx.items():
			if i < self.MAX_NUM_WORDS:
				embedding_vector = word2vec.get(word)
				if embedding_vector is not None:
					self.embedding_matrix[i] = embedding_vector

	def load_decoder_targets_one_hot(self,):
		self.decoder_targets_one_hot = np.zeros(
					(
						len(self.input_texts),
						self.MAX_LEN_TARGET,
						self.NUM_WORDS),
						dtype='float32'
					)
		print ("Decoder one hot shape:",self.decoder_targets_one_hot.shape)

		for i,d in enumerate(self.decoder_targets):
			for t,word in enumerate(d):
				if word != 0:
					self.decoder_targets_one_hot[ i, t, word] = 1

	def loadmodel(self,):
		print ("Model loaded")
		self.TrainModel = CHATBOT_MODEL(self.MAX_LEN_INPUT,self.MAX_LEN_TARGET,self.EMBEDDING_DIM,self.LATENT_DIM,self.NUM_WORDS)
		self.TrainModel.LoadModel(self.embedding_matrix,self.NUM_WORDS)

	def fitmodel(self,BATCH_SIZE=64,EPOCHS=40):
		self.load_decoder_targets_one_hot()

		self.TrainModel.fitModel(self.encoder_inputs,
					self.decoder_inputs,
					self.decoder_targets_one_hot,
					BATCH_SIZE,
					EPOCHS,)
		print ('Done training...')
	def predict(self,):
		self.TrainModel.predict(pad_sequences, self.tokenizer, self.word2idx, self.MAX_LEN_INPUT,)

def main():
	bot = BOT()
	count = 0
	vocabfile = '../large_files/vocab.txt'
	datasetfilename = '../large_files/robot_human.txt'
	while True:
		bot.load_dataset(datasetfilename)
		bot.preprocess_data(vocabfile)
		if count == 0:
			bot.process_pretrained()
			bot.loadmodel()
		bot.fitmodel(BATCH_SIZE=32,EPOCHS=100)
		count += 1
		datasetfilename = input("Another filename to train on(press enter to escape): ")
		if not datasetfilename:
			break


	ans = input("Do you want to try some predictions:")
	if ans.lower().startswith('y'):
		bot.predict()

if __name__ == "__main__":
	main()
