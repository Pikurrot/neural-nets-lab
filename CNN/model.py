import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(torch.nn.Module):
	def __init__(self):
		super(CNN, self).__init__()				# (b, 1, 28, 28)
		self.conv1 = nn.Conv2d(1, 16, 3, 1) 	# (b, 16, 26, 26)
		self.pool1 = nn.MaxPool2d(2, 2)			# (b, 16, 13, 13)
		self.conv2 = nn.Conv2d(16, 32, 3, 1)	# (b, 32, 11, 11)
		self.conv3 = nn.Conv2d(32, 64, 3, 1)	# (b, 64, 9, 9)
		self.pool2 = nn.MaxPool2d(2, 2)			# (b, 64, 4, 4)
		self.fc1 = nn.Linear(64*4*4, 128)		# (b, 128)
		self.fc2 = nn.Linear(128, 10)			# (b, 10)

	def forward(self, x: torch.Tensor):
		x = F.relu(self.conv1(x))
		x = self.pool1(x)
		x = F.dropout2d(F.relu(self.conv2(x)), p=0.25)
		x = F.dropout2d(F.relu(self.conv3(x)), p=0.25)
		x = self.pool2(x)
		x = x.view(-1, 64*4*4)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x		
