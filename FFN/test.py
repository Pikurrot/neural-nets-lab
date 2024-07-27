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

testset = torchvision.datasets.FashionMNIST(root=DATA_PATH, train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = FFN(784, 128, 10).to(device)
model.load_state_dict(torch.load("model.pt"))
model.eval()

correct = 0
total = 0
with torch.no_grad():
	for data in testloader:
		inputs, labels = data
		inputs, labels = inputs.to(device), labels.to(device)
		inputs = inputs.view(inputs.shape[0], -1)

		outputs = model(inputs)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")
