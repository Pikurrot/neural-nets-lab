import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Optional

class GPT(nn.Module):
	def __init__(
			self,
			vocab_size: int,
			hidden_dim: int,
			n_layers: int,
			n_heads: int,
			max_seq_len: int = 512,
			dropout: float = 0.0
	):
		super(GPT, self).__init__()
		self.embedding = nn.Embedding(vocab_size, hidden_dim)
		self.positional_encoding = nn.Embedding(max_seq_len, hidden_dim)
		self.decoder_layers = nn.ModuleList([
			nn.TransformerDecoderLayer(
				d_model = hidden_dim,
				nhead = n_heads,
				dim_feedforward = hidden_dim * 4,
				dropout = dropout,
				batch_first = True)
			for _ in range(n_layers)
		])
		self.fc = nn.Linear(hidden_dim, vocab_size)

	def forward(
			self,
			x: torch.Tensor,
			lookahead_mask: Optional[torch.Tensor] = None,
			padding_mask: Optional[torch.Tensor] = None
	):
		positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
		x = self.embedding(x) + self.positional_encoding(positions)
		for layer in self.decoder_layers:
			x = layer(x, x, tgt_mask=lookahead_mask, tgt_key_padding_mask=padding_mask)
		x = self.fc(x)
		return x

	def generate_lookahead_mask(self, size: int):
		# result is a matrix with the upper right triangle set to 1 and rest to 0
		# diagonal=1 so we don't include the diagonal
		mask = torch.triu(torch.ones(size, size), diagonal=1)
		return mask # (size, size)

	def generate_padding_mask(self, x: torch.Tensor, pad_token: int):
		mask = x == pad_token
		return mask # (b, seq_len)


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
		if char2idx is not None:
			self.char2idx = char2idx
		else:
			self.chars = list(set(full_text))
			self.chars.append("<PAD>")
			self.char2idx = {c: i for i, c in enumerate(self.chars)}
		self.idx2char = {i: c for c, i in self.char2idx.items()}
		self.vocab_size = len(self.char2idx)
		self.x, self.y = self.get_data()

	def get_data(self):
		x, y = [], []
		for i in range(len(self.text) - self.seq_len):
			x.append([self.char2idx[c] for c in self.text[i:i+self.seq_len]])
			y.append([self.char2idx[c] for c in self.text[i+1:i+self.seq_len+1]])
		return torch.tensor(x), torch.tensor(y)
	
	def __len__(self):
		return len(self.text) - self.seq_len
	
	def __getitem__(self, idx: int):
		assert ((idx >= 0) and (idx < len(self.x))), "Index out of range"
		return self.x[idx], self.y[idx]
	
	def save_mapping(self, path: str):
		torch.save(self.char2idx, path)

	def load_mapping(self, path: str):
		self.char2idx = torch.load(path)
		self.idx2char = {i: c for c, i in self.char2idx.items()}
		self.vocab_size = len(self.char2idx)
