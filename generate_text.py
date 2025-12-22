import torch
from rnn import RNN
from process_data import getVocab

def generate_text(seed, vocab, model, idx2word):
    model.eval()
    seq_len = 5

    generated = [vocab[w] for w in seed]

    for _ in range(100):
        input_seq = generated[-seq_len:]
        x = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)

        logits = model(x)

        logits = logits[:, -1] if logits.dim() == 3 else logits

        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_idx)

    return " ".join(idx2word[i] for i in generated)
    
if __name__ == '__main__':
    vocab, _, idx2word = getVocab()
    model = RNN(len(vocab), embed_size=128, hidden_size=256)
    state_dict = torch.load("models/rnn_weights.pth")
    model.load_state_dict(state_dict)

    print(generate_text(["to", "be", "or", "not", "to"], vocab, model, idx2word))