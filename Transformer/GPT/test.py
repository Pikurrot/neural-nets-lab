import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dotenv
import os
import numpy as np
from model import GPT, ShakespeareDataset

dotenv.load_dotenv()
DATA_PATH = os.path.join(os.getenv("DATA_PATH"), "shakespeare.txt")

# Hyperparameters
seq_len = 256
batch_size = 256
embedding_dim = 120
hidden_dim = 384
n_layers = 6
n_heads = 6
dropout = 0.2
lr = 3e-4
epochs = 10

# Dataset and DataLoader
print("Loading data...")
testset = ShakespeareDataset(DATA_PATH, seq_len, train=False, train_frac=0.8)
testset.save_mapping("mapping.pt")
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
print("Testset size:", len(testset))
print("Testloader size:", len(testloader))

# Model
vocab_size = testset.vocab_size
pad_token = testset.char2idx["<PAD>"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = GPT(vocab_size, hidden_dim, n_layers, n_heads, dropout=dropout).to(device)
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Text generation
text = "ROMEO:"
next_chars = 1000

chars = [char for char in text]
x = torch.tensor([[testset.char2idx[char] for char in chars]]).to(device)  # (1, text_len)
if x.size(1) < seq_len:
	# Pad sequence if it's shorter than seq_len
	pad_size = seq_len - x.size(1)
	x = F.pad(x, (pad_size, 0), value=pad_token)

with torch.no_grad():
	for _ in range(next_chars):
		lookahead_mask = model.generate_lookahead_mask(x.size(1)).to(device)
		padding_mask = model.generate_padding_mask(x, pad_token).to(device)
		y_pred = model(x, lookahead_mask, padding_mask)
		last_char_logits = y_pred[0, -1, :]
		p = F.softmax(last_char_logits, dim=0).cpu().numpy()
		# char_idx = np.random.choice(len(last_char_logits), p=p)
		char_idx = np.argmax(p)
		chars.append(testset.idx2char[char_idx])
		
		x = torch.cat([x, torch.tensor([[char_idx]], device=device)], dim=1)
		if x.size(1) > seq_len:
			x = x[:, -seq_len:]

print("Example prediction:")
print("".join(chars))
