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
trainset = ShakespeareDataset(DATA_PATH, seq_len, train=True, train_frac=0.8)
trainset.save_mapping("mapping.pt")
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
print("Trainset size:", len(trainset))
print("Trainloader size:", len(trainloader))

# Model
vocab_size = trainset.vocab_size
pad_token = trainset.char2idx["<PAD>"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = GPT(vocab_size, hidden_dim, n_layers, n_heads, dropout=dropout).to(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training
losses = []

lookahead_mask = model.generate_lookahead_mask(seq_len).to(device)
for epoch in range(epochs):
	model.train()
	for i, (x, y) in enumerate(trainloader):
		x, y = x.to(device), y.to(device)  # (b, seq_len), (b, seq_len)

		optimizer.zero_grad()
		padding_mask = model.generate_padding_mask(x, pad_token).to(device)
		logits = model(x, lookahead_mask, padding_mask)  # (seq_len, b, vocab_size)
		loss = criterion(logits.view(-1, vocab_size), y.view(-1))
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
		if i % 100 == 0:
			print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")

			# Example generation
			model.eval()
			with torch.no_grad():
				start = np.random.randint(0, len(trainset.text) - seq_len)
				input_seq = trainset.text[start:start+seq_len]
				print("-> Input sequence:", input_seq)
				generated = []
				x_gen = torch.tensor([[trainset.char2idx[c] for c in input_seq]]).to(device)

				for j in range(10):
					lookahead_mask_gen = model.generate_lookahead_mask(x_gen.size(1)).to(device)
					padding_mask_gen = model.generate_padding_mask(x_gen, pad_token).to(device)
					logits = model(x_gen, lookahead_mask_gen, padding_mask_gen) # (b, seq_len, vocab_size)
					if j == 0:
						ps = F.softmax(logits, dim=2)
						chars_idx = torch.argmax(ps, dim=2).cpu().numpy()
						print("-> Generated sequence:")
						print("".join([trainset.idx2char[c] for c in chars_idx[0]]).replace("\n", "\\n"))
					last_char_logits = logits[0, -1, :]
					p = F.softmax(last_char_logits, dim=0)
					char_idx = torch.multinomial(p, num_samples=1)
					generated.append(trainset.idx2char[char_idx.item()])
					x_gen = torch.cat([x_gen, char_idx], dim=1)
					if x_gen.size(1) > seq_len:
						x_gen = x_gen[:, -seq_len:] # Keep only the last seq_len characters

				print("-> Generated continuation:", ''.join(generated).replace("\n", "\\n"))
			model.train()
	print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), "model.pt")

print("Done! Train loss:", losses[-1])
