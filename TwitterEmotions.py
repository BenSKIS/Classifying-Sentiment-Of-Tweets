import numpy as numpy
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras_nlp
import keras
from keras_nlp import models

import transformers
import seaborn as sns
import matplotlib.pyplot as plt

text = pd.read_csv('text.csv')
text.shape
text.head()

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text
texts = list(text['text'])
tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
t_text = tokenized_texts

labels = keras.utils.to_categorical(text['label'])
df = tf.data.Dataset.from_tensor_slices((dict(t_text),labels))

split = .3
total = len(texts)
train_size = int((1-split)*len(texts))
test_size = len(texts) - train_size
train_dataset = df.take(train_size).batch(4).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = df.skip(train_size).batch(4).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Define the model
with tf.device('/CPU:0'):
    model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(np.unique(text['label'])))

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
optimizer = mixed_precision.LossScaleOptimizer(optimizer)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model
with tf.device('/CPU:0'):
    model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(np.unique(text['label'])))

model.fit(train_dataset, epochs=3, validation_data=test_dataset)
    
























































model_name1 = 'microsoft/deberta-v3-base'
tokenizer = AutoTokenizer.from_pretrained(model_name1)
classifier2 = DebertaV2ForSequenceClassification.from_pretrained(model_name1, num_labels=6)
classifier2.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])


from transformers import DebertaV2ForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
import torch

model_name1 = "microsoft/deberta-v3-base" # or any other model
tokenizer = AutoTokenizer.from_pretrained(model_name1)
model = DebertaV2ForSequenceClassification.from_pretrained(model_name1, num_labels=3)

optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()


model.train() # Set the model to training mode

for epoch in range(num_epochs):
    for batch in train_dataloader: # Assuming you have a DataLoader
        optimizer.zero_grad() # Clear previous gradients
        inputs = batch['input_ids'].to(device) # Move inputs to device (CPU/GPU)
        labels = batch['labels'].to(device) # Move labels to device
        
        outputs = model(inputs, labels=labels) # Forward pass
        loss = outputs.loss # Extract the loss
        loss.backward() # Compute gradients
        optimizer.step() # Update model parameters
