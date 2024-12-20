import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image, ImageDraw, ImageFont

images_path = 'Handwritten_Text_Generator\HKR_Dataset-master\HKR_Dataset-master\images'
IMG_HEIGHT = 64
IMG_WIDTH = 64
image_files = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_data = []
text_data = []

for img_file in image_files:
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_normalized = img_resized / 255.0
    image_data.append(img_normalized)
    label = os.path.splitext(os.path.basename(img_file))[0]
    text_data.append(label)

image_data = np.array(image_data)

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(text_data)
sequences = tokenizer.texts_to_sequences(text_data)
input_sequences = []
output_sequences = []

for seq in sequences:
    for i in range(1, len(seq)):
        input_sequences.append(seq[:i])
        output_sequences.append(seq[i])

max_seq_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length)
vocab_size = len(tokenizer.word_index) + 1
output_sequences = to_categorical(output_sequences, num_classes=vocab_size)

embedding_dim = 128
hidden_units = 256

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length),
    LSTM(hidden_units, return_sequences=False),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

EPOCHS = 30
BATCH_SIZE = 64
history = model.fit(input_sequences, output_sequences, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

def generate_text(model, start_string, tokenizer, num_chars=100):
    input_text = start_string
    for _ in range(num_chars):
        tokenized_input = tokenizer.texts_to_sequences([input_text])
        tokenized_input = pad_sequences(tokenized_input, maxlen=max_seq_length)
        predictions = model.predict(tokenized_input, verbose=0)
        predicted_index = np.argmax(predictions[0])
        predicted_char = tokenizer.index_word.get(predicted_index, "")
        input_text += predicted_char
    return input_text

start_string = "hello"
generated_text = generate_text(model, start_string, tokenizer)
print("Generated Text:", generated_text)

def render_text_as_image(text, output_path="output_image.png"):
    img = Image.new('L', (800, 200), color=255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", size=32)
    draw.text((10, 50), text, font=font, fill=0)
    img.show()
    img.save(output_path)
    print("Rendered image saved to:", output_path)

render_text_as_image(generated_text)
