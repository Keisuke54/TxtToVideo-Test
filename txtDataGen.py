# text input
inputTxt = ["waterfall from bird's eye view in a forest with moisture",
               "a surfer being covered by backlight and shaded dark looking at shore in orange sunset",
               "a woman on a left and a man on a right working together",
               "a white car moving across a forest road",
               "strawberries in a busket with unstable camera focus",
               "aerial footage an around rocky cliff covered with green in volcanic area under overcast weather",
               "zooming out of deadsea shore, white land and green sea",
               "black and white footage of hands turning to the next page of a book in a shadow",
               "a man in a red shirt & black pants and a man in a blue jacket & black shirt & black pants standing by an abandoned airplane on the shore",
               "a person wearing jeans swinging crossed foot with orange sox and a white shoe labled as TRAVEL",
               "aeriala timelapse over a town with lots of trees, roads and cars in overcast weather",
               "a woman in a white shirt and brown skirt kissing with a man in brown jacket, blueshirt",
               "aerial footage of three sailboats in ocaen",
               "dancers and musicians performing on a promenade",
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

for x in range(len(inputTxt)):
    sequence = tokenizer.texts_to_sequences(inputTxt)[x]
    sequence = sequence[:max_len]
    for i in range(max_len - len(sequence)):
        sequence.append(0)
    txtSequence.append(sequence)
    
txtSequence = np.array(txtSequence)

with open('txtData.py', 'w') as file:
    file.write("txtSequence = [\n")
    for sublist in txtSequence:
        file.write("    [")
        file.write(", ".join(str(x) for x in sublist))
        file.write("],\n")
    file.write("]")

