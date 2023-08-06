# imports 
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Embedding, MaxPooling2D, Conv2D, Flatten, Reshape

# text data
from txtData import txtSequence
txtSequence = np.array(txtSequence)

# video data
vidSequence = np.load("vidData.npy")

model = Sequential()
model.add(Embedding(input_dim=100, output_dim=32, input_length=100))
model.add(Reshape((100, 32, 1)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same',input_shape=(100, 224, 224)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(100 * 224 * 224, activation='sigmoid'))
model.add(Reshape((100, 224, 224)))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary

model.fit(txtSequence, vidSequence, epochs=10, batch_size=32)

model.save('/kaggle/working/vidGenModel.h5')
