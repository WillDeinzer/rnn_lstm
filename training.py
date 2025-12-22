import torch
import torch.nn as nn
from rnn import RNN
from process_data import getDataloader
from tqdm import tqdm

def train_rnn(num_epochs, vocab_size, dataloader):
    model = RNN(vocab_size, embed_size=128, hidden_size=256)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(num_epochs), desc="Training RNN..."):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()
        
        print(f"Epoch {epoch + 1} loss: {total_loss}")

    PATH = 'models/rnn_weights.pth'
    torch.save(model.state_dict(), PATH)
    

if __name__ == '__main__':
    dataloader, vocab_size = getDataloader()
    train_rnn(5, vocab_size, dataloader)