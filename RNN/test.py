import torch
from torch.utils.data import DataLoader
import dotenv
import os
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

print("Some example predictions:")
for i in range(1):
	print("-> Input:")
	print("".join([testset.idx2char[j.item()] for j in x[i]]))
	print("-> Ground truth:")
	print("".join([testset.idx2char[j.item()] for j in y[i]]))
	print("-> Prediction:")
	print("".join([testset.idx2char[j.item()] for j in y_pred[i]]))
	print()
