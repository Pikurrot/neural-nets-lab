import torch
import torchvision
from torch.utils.data import DataLoader
import dotenv
import os
from model import FFN

dotenv.load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")

transform = torchvision.transforms.Compose([
	torchvision.transforms.ToTensor(),
	torchvision.transforms.Normalize((0.5,), (0.5,)) # (x - 0.5) / 0.5
])

batch_size = 64
input_dim = 784
hidden_dim = 128
output_dim = 10
lr = 0.001

trainset = torchvision.datasets.FashionMNIST(root=DATA_PATH, train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = FFN(input_dim, hidden_dim, output_dim).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

losses = []
for epoch in range(5):
	for i, (x, y) in enumerate(trainloader):
		x, y = x.to(device), y.to(device) # (b, 1, 28, 28), (b)
		x = x.view(x.shape[0], -1) # (b, 784)

		optimizer.zero_grad()
		logits = model(x) # (b, 10)
		loss = criterion(logits, y)
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
		if i % 100 == 0:
			print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")
	print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), "model.pt")

print("Done! Train loss:", losses[-1])		
