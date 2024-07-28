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

batch_size = 64
input_dim = 784
hidden_dim = 128
output_dim = 10

testset = torchvision.datasets.FashionMNIST(root=DATA_PATH, train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = FFN(input_dim, hidden_dim, output_dim).to(device)
model.load_state_dict(torch.load("model.pt"))
model.eval()

correct = 0
total = 0
with torch.no_grad():
	for (x, y) in testloader:
		x, y = x.to(device), y.to(device) # (b, 1, 28, 28), (b)
		x = x.view(x.shape[0], -1) # (b, 784)

		logits = model(x) # (b, 10)
		_, y_pred = torch.max(logits, 1) # (b,)
		total += y.size(0)
		correct += (y_pred == y).sum().item()

print(f"Accuracy: {100 * correct / total}%")
