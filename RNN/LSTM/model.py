import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Optional

class LSTM(torch.nn.Module):
	def __init__(
			self,
			input_dim: int,
			embedding_dim: int,
			hidden_dim: int,
			output_dim: int,
			n_layers: int = 1,
			drop_prob: float = 0.0,
			bidirectional: bool = False
	):
		super(LSTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = n_layers
																	
		self.embedding = nn.Embedding(input_dim, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True,
					dropout=drop_prob, bidirectional=bidirectional)
		self.fc = nn.Linear(hidden_dim, output_dim)

	def init_hidden(self, batch_size: int):
		hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim) # (num_layers, b, hidden_dim)
		cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim) # (num_layers, b, hidden_dim)
		return hidden_state, cell_state
	
	def forward(self, x, hc):			# (b, seq_len)
		embeds = self.embedding(x)		# (b, seq_len, embedding_dim)
		out, hc = self.lstm(embeds, hc)	# (b, seq_len, hidden_dim), (num_layers, b, hidden_dim)
		out = self.fc(out)				# (b, seq_len, output_dim)
		return out, hc


class ShakespeareDataset(Dataset):
	def __init__(
			self,
			data_path: str,
			seq_len: int,
			train: bool = True,
			train_frac: float = 0.8,
			char2idx: Optional[dict] = None
	):
		self.seq_len = seq_len
		with open(data_path, "r") as f:
			full_text = f.read()
		if train:
			self.text = full_text[:int(train_frac * len(full_text))]
		else:
			self.text = full_text[int(train_frac * len(full_text)):]
		self.chars = list(set(full_text))
		if char2idx is not None:
			self.char2idx = char2idx
		else:
			self.char2idx = {c: i for i, c in enumerate(self.chars)}
		self.idx2char = {i: c for c, i in self.char2idx.items()}
		self.vocab_size = len(self.char2idx)
		self.x, self.y = self.get_data()

	def get_data(self):
		x, y = [], []
		for i in range(0, len(self.text) - self.seq_len, self.seq_len):
			x.append([self.char2idx[c] for c in self.text[i:i+self.seq_len]])
			y.append([self.char2idx[c] for c in self.text[i+1:i+self.seq_len+1]])
		return torch.tensor(x), torch.tensor(y)
	
	def __len__(self):
		return len(self.text) // self.seq_len
	
	def __getitem__(self, idx: int):
		assert ((idx >= 0) and (idx < len(self.x))), "Index out of range"
		return self.x[idx], self.y[idx]
	
	def save_mapping(self, path: str):
		torch.save(self.char2idx, path)

	def load_mapping(self, path: str):
		self.char2idx = torch.load(path)
		self.idx2char = {i: c for c, i in self.char2idx.items()}
		self.vocab_size = len(self.char2idx)
