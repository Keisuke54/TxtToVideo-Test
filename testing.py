import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

from keras.models import load_model

model = load_model('vidGenModel.h5', compile=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# text input
inputTxt = [
    "green river from bird's point of view with a man in orange clothing",
    ]

import tensorflow as tf
import numpy as np

txtSequence = []

Tokenizer = tf.keras.preprocessing.text.Tokenizer
tokenizer = Tokenizer(char_level=True, lower=True)

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

tokenizer.fit_on_texts(inputTxt)
vocab_size = len(tokenizer.word_index) + 1

max_len = 100

sequence = tokenizer.texts_to_sequences(inputTxt)
sequence = pad_sequences(sequence, maxlen=max_len, padding='post')

sequence = np.array(sequence)

result = model.predict(sequence)
result = result[0]

print(result)
print(result.shape)

frames = []
fig = plt.figure()
for i in range(len(result)):
    frames.append([plt.imshow(result[i], cmap=cm.Greys_r,animated=True)])

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                repeat_delay=1000)
ani.save('genMovie.mp4')
plt.show()