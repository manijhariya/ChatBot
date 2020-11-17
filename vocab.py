### here it is
## 1.cornell_cleaned.txt
## 2.robot_human.txt
## 3.twitter_tab_format
## 4.GeneralTalk
from keras.preprocessing.text import Tokenizer

MAX_NUM_WORDS = 10000

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

def data(filepath):
	inputs = []
	outputs = []
	for i in open(filepath):
		inputline, output, *_ = i.rstrip().split('\t')
		if len(inputline) < 40 and len(output) < 40:
			inputs.append(inputline)
			outputs.append(output)
			continue

		#print (inputline+ "\t" + output)
		#input()

	return inputs+outputs

rh = data('../large_files/robot_human.txt')
twitter_tab = data('../large_files/twitter_tab_format.txt')
generaltalk = data('../large_files/GeneralTalk.txt')
comicstalk = data('../large_files/ComicsTalk.txt')
cornell = data('../large_files/cornell_cleaned.txt')
"""
tokenizer.fit_on_texts(rh)
word2idx = tokenizer.word_index
print ("after robot human %d"%len(word2idx))

tokenizer.fit_on_texts(generaltalk)
word2idx = tokenizer.word_index
print ("after general talk %d"%len(word2idx))

tokenizer.fit_on_texts(comicstalk)
word2idx = tokenizer.word_index
print ("after comics talk %d"%len(word2idx))

tokenizer.fit_on_texts(generaltalk)
word2idx = tokenizer.word_index
print ("after twittertalk %d"%len(word2idx))

tokenizer.fit_on_texts(cornell)
word2idx = tokenizer.word_index
print ("after cornell %d"%len(word2idx))

with open('../large_files/vocab.txt','w') as f:
	f.write('<eos>\n')
	f.write('<sos>\n')
	i = 2
	for k,v in word2idx.items():
		if i > MAX_NUM_WORDS:
			break
		f.write(k.rstrip()+'\n')
		i += 1
"""
#print ("Vocab is ready and it has %d words"%i)
print ("Total line are %d"%(len(rh) + len(generaltalk) + len(comicstalk)))
