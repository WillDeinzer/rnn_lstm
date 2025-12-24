import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size

        # For LSTM, we would normally need 8 different matrices for h_t-1 and x_t. But we can just use one
        self.W_x = nn.Linear(embed_size, 4 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size)

        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len)
        hidden: (h_0, c_0)
        """
        batch_size, seq_len = x.shape

        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h, c = hidden

        for t in range(seq_len):
            x_t = self.embedding(x[:, t])

            gates = self.W_x(x_t) + self.W_h(h)
            f, i, g, o = gates.chunk(4, dim=1)

            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            g = torch.tanh(g)
            o = torch.sigmoid(o)

            c = f * c + i * g
            h = o * torch.tanh(c)

        logits = self.fc_out(h)
        return logits, (h, c)


