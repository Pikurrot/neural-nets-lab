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
	torchvision.transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.FashionMNIST(root=DATA_PATH, train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = FFN(784, 128, 10).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

losses = []
for epoch in range(5):
	for i, data in enumerate(trainloader):
		inputs, labels = data
		inputs, labels = inputs.to(device), labels.to(device)
		inputs = inputs.view(inputs.shape[0], -1)

		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
		if i % 100 == 0:
			print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")
	print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), "model.pt")

print("Done! Train loss:", losses[-1])		
