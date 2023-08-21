import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the previously saved trained model and tokenizer
model_path = 'sentiment_analysis_trained_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# Take input from the user
input_text = input("Enter a movie review: ")

# Preprocess the input text
inputs = tokenizer(input_text, padding=True, truncation=True, max_length=256, return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Use the trained model to predict sentiment
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

# Display the predicted sentiment
sentiment = "Positive" if predicted_class == 1 else "Negative"
print(f"Predicted Sentiment: {sentiment}")
print(f"Input Text: {input_text}")
