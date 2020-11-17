import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from botmodel import CHATBOT_MODEL

class PredictModel:
	def __init__(self,MAX_VOCAB=10000,max_len_input=40,max_len_target=40,embedding_dim=100,latent_dim=256,max_num_words=10000):
		self.MAX_VOCAB = MAX_VOCAB
		self.MAX_LEN_INPUT = max_len_input
		self.MAX_LEN_TARGET = max_len_target
		self.EMBEDDING_DIM = embedding_dim
		self.LATENT_DIM  = latent_dim
		self.MAX_NUM_WORDS = max_num_words

	def load_model(self,):
		self.model = CHATBOT_MODEL(self.MAX_LEN_INPUT, self.MAX_LEN_TARGET,self.EMBEDDING_DIM, self.LATENT_DIM,self.MAX_VOCAB)
		embedding_matrix = np.zeros((self.MAX_VOCAB,self.EMBEDDING_DIM))
		self.model.LoadModel(embedding_matrix,self.MAX_VOCAB)

	def load_vocab(self,vocab_filename):
		self.vocab = []
		for i in open(vocab_filename):
			self.vocab.append(i.rstrip())
		tokenizer = Tokenizer(self.MAX_VOCAB,filters='')
		tokenizer.fit_on_texts(self.vocab)
		word2idx = tokenizer.word_index
		return tokenizer,word2idx

	def predict(self,vocab_filename,weightfile):
		self.load_model()
		tokenizer, word2idx = self.load_vocab(vocab_filename)
		self.model.predict(pad_sequences, tokenizer, word2idx, self.MAX_LEN_INPUT,weightfile=weightfile)

def main():
	Pmodel = PredictModel()
	weightfile = 'CHATBOT_MODELweight.h5'
	Pmodel.predict('../large_files/vocab.txt',weightfile)

if __name__ == "__main__":
	main()
