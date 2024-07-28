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
		super(FFN, self).__init__()						
		self.fc1 = nn.Linear(input_dim, hidden_dim)		
		self.fc2 = nn.Linear(hidden_dim, output_dim)

	def forward(self, x: torch.Tensor):	# (b, input_dim)
		x = F.relu(self.fc1(x))			# (b, hidden_dim)
		x = self.fc2(x)					# (b, output_dim)
		return x
