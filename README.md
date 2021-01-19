## Introduction
* A Chatter bot which talk mostly like me
* Collected data from [Movie-Dialogue](https://www.kaggle.com/rajathmc/cornell-moviedialog-corpus)
* Engineered Movie dialogues to covert them into a 2-way human talk like structure
* Engineered preprocessed those Talks into data which can be suitable to fit in model 
* Build a Encoder and Decoder using Recurrent Neural Network layers to train model.
* Build a Encoder and Decoder using Recurrent Neural Network layers to use in API tasks
* Build a terminal facing API to make it talk.
<br>
<p align="center">
  <img src="static/temp.jpeg" width="450" alt="accessibility text">
</p>
<br>

## Code and Resources Used

Python Version: 3.6
Packages: pandas,numpy,matplotlib,tensorflow,

## Data Collecting
Did some research and found the best dataset for spoken digits in [Movie-Dialogues](https://www.kaggle.com/rajathmc/cornell-moviedialog-corpus)

## Data Cleaning and EDA
After downloading the data, Scripted in python to clean the data so that it can be used by model.
    * Converted Movie dialogues into simple human talk structure
    * Build a dictionary using all words present in those Sentences
    * Preprcoessed data to convert them each sentence into sentence of length less than 40
    * Tokenized every sentence to get word2idx 
    * Added a padding for sentences have length less than 40
    * Added a token at the end of every sentence to remind ChatBot.
    * Pretained the data using [Glove Vectors](https://nlp.stanford.edu/projects/glove/) for word representation
    * Build a Decoder One Hot Key targets for output

## Model Building
Data was ready to fit into the model.
Using Neural Network
   * Implimented model in Tensorflow2 (keras) in Recurrent Neural Network approach
   
   ** Encoder

    Data ---> Embedded layer ---> (Encoder)LSTM layer ---> Encoder Ouput
   ** Decoder
   
    Encoder Ouput ---> (Decoder)LSTM layers ----> Dense layer ----> Decoder Output
   * Build a custom loss and accuracy function
   * Compiled Model with Adam Optimizer with default learning rate
   * With this model i ended up with 91.2 % accuracy by metrics


## Productionization
  * Build a idx2word tokenizer for every word2idx token
  * In this task i have built a different decoder sequence than the train model sequence
  * Build a different class to load the saved model weights and use them to predict on future sentences.
  * Build a simple terminal interface to work with ChatBot

## Conclusion
  Neural Networks specially LSTM/GRU layers for network building are most powerful and used layers for AI task. This simple
  Encoder/Decoder model can make do more talking like humans and can also be used as a query solution in large production companies. 
