#!/usr/bin/python3

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from compile_from_xsid import compile_from_xsid

class DialectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data(path):
    df = compile_from_xsid(path)
    texts = df["text"].tolist()
    labels = pd.factorize(df["intent"])[0]
    return texts, labels


texts, labels = load_data("../data/de-by.test.conll")
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
train_dataset = DialectDataset(train_texts, train_labels, tokenizer)
val_dataset = DialectDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(set(labels)))
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def train(model, train_loader, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss, correct = 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.logits.argmax(1) == labels).sum().item()
        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader)}, Accuracy = {correct / len(train_loader.dataset)}")


def evaluate(model, val_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            correct += (outputs.logits.argmax(1) == labels).sum().item()
    print(f"Validation Accuracy: {correct / len(val_loader.dataset)}")


train(model, train_loader)
evaluate(model, val_loader)


# model.save_pretrained("mbert-dialect-classifier")
# tokenizer.save_pretrained("mbert-dialect-classifier")
