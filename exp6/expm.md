from keras.models import Model
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample data
input_texts = ['I love NLP', 'He plays football']
target_texts = [['PRON', 'VERB', 'NOUN'], ['PRON', 'VERB', 'NOUN']]

# Vocabulary
word_vocab = sorted(set(word for sent in input_texts for word in sent.split()))
tag_vocab = sorted(set(tag for tags in target_texts for tag in tags))

word2idx = {word: i + 1 for i, word in enumerate(word_vocab)}  # start from 1
tag2idx = {tag: i for i, tag in enumerate(tag_vocab)}

# Parameters
max_len = max(len(sent.split()) for sent in input_texts)
num_words = len(word2idx) + 1
num_tags = len(tag2idx)

# Prepare input and output
X = [[word2idx[word] for word in sent.split()] for sent in input_texts]
X = pad_sequences(X, maxlen=max_len, padding='post')

y = [[tag2idx[tag] for tag in tags] for tags in target_texts]
y = pad_sequences(y, maxlen=max_len, padding='post')
y = np.expand_dims(y, -1)  # required for sparse_categorical_crossentropy

# Model
input_layer = Input(shape=(max_len,))
embedding = Embedding(input_dim=num_words, output_dim=64)(input_layer)
lstm = LSTM(64, return_sequences=True)(embedding)
output = TimeDistributed(Dense(num_tags, activation='softmax'))(lstm)

model = Model(input_layer, output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
<img width="570" height="245" alt="Screenshot 2025-10-08 105924" src="https://github.com/user-attachments/assets/03b21fad-6c32-4256-9838-f3ddeed72d62" />
