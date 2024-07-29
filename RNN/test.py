import torch
from torch.utils.data import DataLoader
import dotenv
import os
import numpy as np
from model import RNN, ShakespeareDataset

dotenv.load_dotenv()
DATA_PATH = os.path.join(os.getenv("DATA_PATH"), "shakespeare.txt")
	
seq_len = 100
batch_size = 64

testset = ShakespeareDataset(DATA_PATH, seq_len, train=False, train_frac=0.8)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

input_dim = testset.vocab_size
embedding_dim = 300
hidden_dim = 1024
output_dim = testset.vocab_size
n_layers = 2
drop_prob = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = RNN(input_dim, embedding_dim, hidden_dim, output_dim, n_layers, drop_prob).to(device)
model.load_state_dict(torch.load("model.pt"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
	for (x, y) in testloader:
		this_batch_size = x.shape[0]
		state = model.init_hidden(this_batch_size).to(device) # (n_layers, b, embedding_dim)
		x, y = x.to(device), y.to(device) # (b, seq_len), (b, seq_len)

		logits, state = model(x, state) # (b, seq_len, vocab_size), (n_layers, b, embedding_dim)
		_, y_pred = torch.max(logits, 2) # (b, seq_len)
		total += y.size(0) * y.size(1)
		correct += (y_pred == y).sum().item()

print(f"Accuracy: {100 * correct / total}%")

# Text generation
text = "ROMEO:"
next_chars = 1000

chars = [char for char in text]
state = model.init_hidden(1).to(device) # (n_layers, 1, embedding_dim)

x = torch.tensor([testset.char2idx[char] for char in chars]) # (text_len,)
x = x.unsqueeze(0) # (1, text_len)
x = x.to(device)

with torch.no_grad():
	for i in range(0, next_chars):
		y_pred, state = model(x, state)

		last_char_logits = y_pred[0][-1] # (vocab_size)
		p = torch.nn.functional.softmax(last_char_logits, dim=0).detach().cpu().numpy()
		# char_index = torch.argmax(last_char_logits).item()
		char_idx = np.random.choice(len(last_char_logits), p=p)
		chars.append(testset.idx2char[char_idx])

		# the output of the model (a single character index) becomes the input at next iteration
		x = torch.tensor([[char_idx]]) # (1, 1)
		x = x.to(device)

print("Example prediction:")
print("".join(chars))
