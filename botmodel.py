from keras.models import Model
from keras.layers import Embedding,Input,LSTM,Dense
import matplotlib.pyplot as plt
import numpy as np

try:
	import keras.backend as K
	if len(K.tensorflow_backend._get_available_gpus()) > 0:
		from keras.layers import CuDNNLSTM as LSTM
		from keras.layers import CuDNNGRU as GRU
except:
	pass

class CHATBOT_MODEL:
	def __init__(self,max_len_input,max_len_target,embedding_dim,latent_dim,num_words):
		self.MAX_LEN_INPUT = max_len_input
		self.MAX_LEN_TARGET = max_len_target
		self.EMBEDDING_DIM = embedding_dim
		self.LATENT_DIM = latent_dim
		self.NUM_WORDS = num_words

	def LoadModel(self,embedding_matrix,num_words_output):

		embedding_layer = Embedding(
					self.NUM_WORDS,
					self.EMBEDDING_DIM,
					weights=[embedding_matrix],
					input_length=self.MAX_LEN_INPUT,
					trainable=True,
					)

		self.encoder_inputs_placeholder = Input(shape=(self.MAX_LEN_INPUT))
		x = embedding_layer(self.encoder_inputs_placeholder)

		encoder = LSTM(
				self.LATENT_DIM,
				return_state=True,
				)
		encoder_outputs, h, c = encoder(x)
		self.encoder_states = [h,c]

		decoder_inputs_placeholder = Input(shape=(self.MAX_LEN_TARGET))
		self.decoder_embedding = Embedding(num_words_output, self.EMBEDDING_DIM)
		decoder_inputs_x =  self.decoder_embedding(decoder_inputs_placeholder)

		self.decoder_lstm = LSTM(
				self.LATENT_DIM,
				return_state = True,
				return_sequences = True,
				)

		decoder_outputs,_,_ = self.decoder_lstm(
						decoder_inputs_x,
						initial_state = self.encoder_states
						)
		self.decoder_dense = Dense(self.NUM_WORDS,activation='softmax')
		decoder_outputs = self.decoder_dense(decoder_outputs)

		self.model = Model([self.encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)
		self.model.compile(optimizer='adam',
				loss = self.custom_loss,
				metrics=[self.acc])

		#return model

	def custom_loss(self,y_true,y_pred):
		mask = K.cast(y_true > 0,dtype='float32')
		out = mask * y_true * K.log(y_pred)
		return -K.sum(out) / K.sum(mask)

	def acc(self,y_true,y_pred):
		targ = K.argmax(y_true, axis=-1)
		pred = K.argmax(y_pred, axis=-1)
		correct = K.cast(K.equal(targ,pred), dtype='float32')

		mask = K.cast(K.greater(targ,0), dtype='float32')
		n_correct = K.sum(mask * correct)
		n_total = K.sum(mask)
		return n_correct / n_total

	def fitModel(self,encoder_inputs,decoder_inputs,
		decoder_targets_one_hot,BATCH_SIZE,EPOCHS,
		graphs=True,save=True):
		r = self.model.fit(
			[encoder_inputs,decoder_inputs],
			decoder_targets_one_hot,
			batch_size = BATCH_SIZE,
			epochs=EPOCHS,
			validation_split=0.2,
			)
		if save:
			self.model.save('CHATBOT_MODEL.h5')
			self.model.save_weights('CHATBOT_MODELweight.h5')
		if graphs:
			plt.plot(r.history['loss'],label='loss')
			plt.plot(r.history['val_loss'],label='val_loss')
			plt.legend()
			plt.show()

			plt.plot(r.history['acc'],label='acc')
			plt.plot(r.history['val_acc'],label='val_acc')
			plt.legend()
			plt.show()

	def decoder_sequence(self,input_seq):
		states_value = self.encoder_model.predict(input_seq)
		target_seq = np.zeros((1,1))
		target_seq[0,0] = self.word2idx['<sos>']

		eos = self.word2idx['<eos>']
		output_sentence = []
		for _ in range(self.max_len_target):
			output_tokens,h,c = self.decoder_model.predict(
						[target_seq] + states_value
						)
			idx = np.argmax(output_tokens[0,0,:])

			if eos == idx:
				break

			word = ''
			if idx > 0:
				word = self.idx2word[idx]
				output_sentence.append(word)

			target_seq[0,0] = idx
			states_value = [h,c]

		return ' '.join(output_sentence)

	def predict(self,pad_sequences,tokenizer,word2idx,max_len_input=40,max_len_target=40,weightfile=None):
		if weightfile is not None:
			self.model.load_weights(weightfile)

		self.word2idx = word2idx
		self.idx2word = {v:k for k,v in self.word2idx.items()}

		self.max_len_target = max_len_target

		self.encoder_model = Model(self.encoder_inputs_placeholder,self.encoder_states)
		decoder_state_input_h = Input(shape=(self.LATENT_DIM,))
		decoder_state_input_c = Input(shape=(self.LATENT_DIM,))
		decoder_state_inputs = [decoder_state_input_h,decoder_state_input_c]

		decoder_input_single = Input(shape=(1,))
		decoder_input_single_x = self.decoder_embedding(decoder_input_single)

		decoder_outputs,h,c = self.decoder_lstm(
						decoder_input_single_x,
						initial_state=decoder_state_inputs
					)
		decoder_states = [h,c]
		decoder_outputs = self.decoder_dense(decoder_outputs)

		self.decoder_model = Model(
					[decoder_input_single] + decoder_state_inputs,
					[decoder_outputs] + decoder_states
				)
		print ("Enter exit to quit")
		while True:
			input_seq = input("Me-> ")
			if (input_seq.lower() == "exit"):
				break

			input_seq = tokenizer.texts_to_sequences([input_seq])
			input_seq = pad_sequences(input_seq,maxlen=max_len_input)
			#print (input_seq)  ## for testing only
			translation = self.decoder_sequence(input_seq)
			print ('Bot->',translation)
