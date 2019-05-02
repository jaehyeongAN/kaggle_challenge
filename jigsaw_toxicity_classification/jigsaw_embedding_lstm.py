# Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
from keras import models, layers, Model
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

## load data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print(train_df.shape)
print(test_df.shape)

train_df = train_df[['id','comment_text','target']]
# set index
train_df.set_index('id', inplace=True)
test_df.set_index('id', inplace=True)
# y_label
train_y_label = np.where(train_df['target'] >= 0.5, 1, 0)
train_df.drop(['target'], axis=1, inplace=True)

## tokenize
max_words = 100000
max_len = 220

tokenizer = text.Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_df['comment_text'])
# texts_to_sequences
sequences_text_train = tokenizer.texts_to_sequences(train_df['comment_text'])
sequences_text_test = tokenizer.texts_to_sequences(test_df['comment_text'])
# add padding
pad_train = sequence.pad_sequences(sequences_text_train, maxlen=max_len)
pad_test = sequence.pad_sequences(sequences_text_test, maxlen=max_len)

## embedding + LSTM layers
# model define
model = models.Sequential()
model.add(layers.Embedding(max_words, 128, input_length=max_len))
model.add(layers.Bidirectional(layers.LSTM(64, dropout=0.5, recurrent_dropout=0.3, return_sequences=True)))
model.add(layers.Bidirectional(layers.LSTM(64, dropout=0.5, recurrent_dropout=0.3, return_sequences=True)))

model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.BatchNormalization())

model.add(layers.Dense(1, activation='sigmoid'))

# model compile
model.compile(optimizer='adam',
             loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

# keras.callbacks
callbacks_list = [ReduceLROnPlateau(
                        monitor='val_loss', patience=2, factor=0.2),	# val_loss가 patience동안 향상되지 않으면 학습률을 0.2만큼 감소 (new_lr = lr * factor)
                    ModelCheckpoint(
                        filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

history = model.fit(pad_train, train_y_label,
                     epochs=2, batch_size=1024,
                     callbacks=callbacks_list, 
                     validation_split=0.3)
                     
## predict test_set
test_pred = model.predict(pad_test)

sample_result = pd.DataFrame()
sample_result['id'] = test_df.index
sample_result['prediction'] = test_pred

## submit sample_submission.csv
sample_result.to_csv('submission.csv', index=False)