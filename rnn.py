import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size

        self.W_xh = nn.Parameter(torch.randn(embed_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        self.W_hy = nn.Parameter(torch.randn(hidden_size, vocab_size) * 0.01)
        self.b_y = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        """
        batch_size, seq_len = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            # Get the embedding for the current word in the sequence
            x_t = self.embedding(x[:, t])
            h = torch.tanh(x_t @ self.W_xh + h @ self.W_hh + self.b_h)

        y = h @ self.W_hy + self.b_y # Logits for the next word
        return y

