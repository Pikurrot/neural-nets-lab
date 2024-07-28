import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(torch.nn.Module):
	def __init__(
			self,
			input_dim: int,
			hidden_dim: int,
			output_dim: int
	):
		super(FFN, self).__init__()						# (b, input_dim)
		self.fc1 = nn.Linear(input_dim, hidden_dim)		# (b, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, output_dim)	# (b, output_dim)

	def forward(self, x: torch.Tensor):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x
