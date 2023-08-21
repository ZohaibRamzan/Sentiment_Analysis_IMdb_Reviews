import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
# Load IMDb dataset
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
vocab_size = 10000
max_length = 256
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = [x[:max_length] for x in x_train]
x_test = [x[:max_length] for x in x_test]

# preprocessing
# Pad or truncate sequences to a fixed length
x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')

# Convert to PyTorch tensors
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test)

# Set up DataLoader
batch_size = 16
train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2) # two classes

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_training_steps = len(train_dataloader) * 5  # Total steps for 5 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_training_steps)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#sets the model's mode to training
model.train()

# Initialize empty list to store losses for each step
step_losses = []
for step, batch in enumerate(train_dataloader):
    inputs, labels = batch
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    step_losses.append(loss.item())  # Store loss for this step

    if (step + 1) % len(train_dataloader) == 0:  # Print and plot after each epoch
        average_loss = sum(step_losses) / len(step_losses)
        print(f"Step [{step + 1}/{total_training_steps}], Average Loss: {average_loss:.4f}")
        step_losses = []  # Reset losses for the next epoch

# Plot loss for each training step
plt.plot(step_losses, marker='o')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Training Loss per Training Step')
plt.show()

# Save trained model
model.save_pretrained('sentiment_analysis_trained_model')
tokenizer.save_pretrained('sentiment_analysis_trained_model')
