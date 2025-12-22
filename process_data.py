import re
from torch.utils.data import Dataset, DataLoader
import torch

class WordDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
def getVocab():
    with open("data/train.txt", "r", encoding="utf-8") as f:
        text = f.read().lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        words = text.split()

    vocab = {word: idx for idx, word in enumerate(set(words))}
    idx2word = {idx: word for word, idx in vocab.items()}
    return vocab, words, idx2word

def getDataloader():
    vocab, words, _ = getVocab()

    indices = [vocab[word] for word in words]

    seq_len = 5
    X, Y = [], []
    for i in range(len(indices) - seq_len):
        X.append(indices[i:i+seq_len])
        Y.append(indices[i+seq_len])

    dataset = WordDataset(X, Y)
    batch_size=32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, len(vocab)
