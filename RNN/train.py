import torch
from torch.utils.data import DataLoader
import dotenv
import os
from model import RNN, ShakespeareDataset

dotenv.load_dotenv()
DATA_PATH = os.path.join(os.getenv("DATA_PATH"), "shakespeare.txt")
	
seq_len = 100
batch_size = 256

trainset = ShakespeareDataset(DATA_PATH, seq_len, train=True, train_frac=0.8)
trainset.save_mapping("mapping.pt")
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

input_dim = trainset.vocab_size
embedding_dim = 300
hidden_dim = 1024
output_dim = trainset.vocab_size
n_layers = 2
drop_prob = 0.2
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = RNN(input_dim, embedding_dim, hidden_dim, output_dim, n_layers, drop_prob).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

losses = []
for epoch in range(10):
	for i, (x, y) in enumerate(trainloader):
		this_batch_size = x.shape[0] # Last batch may have different size
		state = model.init_hidden(this_batch_size).to(device) # (n_layers, b, embedding_dim)
		x, y = x.to(device), y.to(device) # (b, seq_len), (b, seq_len)

		optimizer.zero_grad()
		logits, state = model(x, state) # (b, seq_len, vocab_size), (n_layers, b, embedding_dim)
		loss = criterion(logits.view(-1, trainset.vocab_size), y.view(-1))
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
		if i % 100 == 0:
			print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")
	print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), "model.pt")

print("Done! Train loss:", losses[-1])