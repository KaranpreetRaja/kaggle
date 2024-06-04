import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Boi u thought I was giving u comments?
accepted = pd.read_csv('accepted-plates.csv')
rejected = pd.read_csv('rejected-plates.csv')

accepted['label'] = 1
rejected['label'] = 0
dataset = pd.concat([accepted, rejected], ignore_index=True)
dataset = dataset[dataset['plate'].notna()]
print(dataset.head())
print(dataset['label'].value_counts())
print(f'length of dataset: {len(dataset)}')
dataset = dataset.sample(frac=0.01, random_state=1)
print(f'length of dataset BOI: {len(dataset)}')
def encode_text(text):
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
    features = last_hidden_states[:, 0, :].numpy()
    return features
X = dataset['plate']
y = dataset['label']
encoded = X.apply(encode_text)
print(f'encoded shape is {encoded.shape}')
encoded_filtered = encoded.loc[dataset.index]
encoded_plate = encoded_filtered.values.reshape(-1)
dataset['encoded_plate'] = encoded_plate
print(dataset.head())
import pickle
with open('dataset.pickle', 'wb') as f:
    pickle.dump(dataset, f)