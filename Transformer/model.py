import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Literal

class ScaledDotProductAttention(nn.Module):
	def __init__(
			self,
			d_model: int,
			d_k: int,
			dropout: float
	):
		super(ScaledDotProductAttention, self).__init__()
		self.d_model = d_model
		self.d_k = d_k # head size
		self.dropout = nn.Dropout(dropout)
		
	def forward(
			self,
			q: torch.Tensor, # (b, n_heads, q_len, d_k)
			k: torch.Tensor, # (b, n_heads, k_len, d_k)
			v: torch.Tensor, # (b, n_heads, k_len, d_v)
			mask: Optional[torch.Tensor] = None
	):
		matmul = q @ k.transpose(-2, -1) # (b, n_heads, q_len, k_len)
		scale = matmul / (self.d_k ** 0.5)
		if mask is not None:
			scale = scale.masked_fill(mask == 0, -1e9) # -1e9 ~ -inf
		attn_scores = F.softmax(scale, dim=-1) # (b, n_heads, q_len, k_len)
		attn_scores = self.dropout(attn_scores)
		matmul2 = attn_scores @ v # (b, n_heads, q_len, d_v)
		return matmul2, attn_scores # (b, n_heads, q_len, d_v), (b, n_heads, q_len, k_len)


class MultiHeadAttention(nn.Module):
	def __init__(
			self,
			n_heads: int,
			d_model: int,
			d_k: int,
			d_v: int,
			dropout: float
	):
		super(MultiHeadAttention, self).__init__()
		self.n_heads = n_heads
		self.d_model = d_model
		self.d_k = d_k # head size
		self.d_v = d_v

		self.q = nn.Linear(d_model, n_heads * d_k, bias=False)
		self.k = nn.Linear(d_model, n_heads * d_k, bias=False)
		self.v = nn.Linear(d_model, n_heads * d_v, bias=False)
		self.attention = ScaledDotProductAttention(d_model, d_k, dropout)
		self.fc = nn.Linear(n_heads * d_v, d_model)
		self.dropout = nn.Dropout(dropout)
		
	def forward(
			self,
			q_origin: torch.Tensor, # (b, q_len, d_model)
			k_origin: torch.Tensor, # (b, k_len, d_model)
			v_origin: torch.Tensor, # (b, k_len, d_model)
			mask: Optional[torch.Tensor] = None # (b, q_len, k_len)
	):
		# q_origin is where the query comes from, not the query itself
		batch_size = q_origin.size(0)
		q_len = q_origin.size(1) # query length (number of tokens)
		k_len = k_origin.size(1)
		# extract queries, keys, and values for all heads at once
		q = self.q(q_origin).view(batch_size, q_len, self.n_heads, self.d_k)
		k = self.k(k_origin).view(batch_size, k_len, self.n_heads, self.d_k)
		v = self.v(v_origin).view(batch_size, k_len, self.n_heads, self.d_v)

		q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # (b, n_heads, q_len, d_k)

		if mask is not None:
			mask = mask.unsqueeze(1) # (b, 1, q_len, k_len), for broadcasting

		# conceptually, the result of the attention mechanism is a transformation
		# of the queries
		q_updated, attn_scores = self.attention(q, k, v, mask) # (b, n_heads, q_len, d_v)

		# concatenate the heads
		concat = q_updated.transpose(1, 2) # (b, q_len, n_heads, d_v)
		concat = concat.contiguous() # ensure tensor is contiguous in memory before view()
		concat = concat.view(batch_size, q_len, -1) # (b, q_len, n_heads * d_v)

		# linear transformation
		output = self.fc(concat) # (b, q_len, d_model)
		output = self.dropout(output)

		return output, attn_scores # (b, q_len, d_model), (b, n_heads, q_len, k_len)


class PointwiseFeedforward(nn.Module):
	def __init__(
			self,
			d_model: int,
			d_ff: int,
			dropout: float
	):
		super(PointwiseFeedforward, self).__init__()
		self.fc1 = nn.Linear(d_model, d_ff)
		self.fc2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)
		
	def forward(
			self,
			x: torch.Tensor # (b, seq_len, d_model)
	):
		x = F.relu(self.fc1(x)) # (b, seq_len, d_ff)
		x = self.fc2(x) # (b, seq_len, d_model)
		x = self.dropout(x)
		return x # (b, seq_len, d_model)


class EncoderLayer(nn.Module):
	def __init__(
			self,
			n_heads: int,
			d_model: int,
			d_k: int,
			d_v: int,
			d_ff: int,
			dropout: float
	):
		super(EncoderLayer, self).__init__()
		self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
		self.ff = PointwiseFeedforward(d_model, d_ff, dropout)
		self.layernorm1 = nn.LayerNorm(d_model)
		self.layernorm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)
		
	def forward(
			self,
			x: torch.Tensor, # (b, seq_len, d_model)
			mask: Optional[torch.Tensor] = None # (b, seq_len, seq_len)
	):
		# self-attention
		attn_output, attn_scores = self.self_attn(x, x, x, mask) # (b, seq_len, d_model)
		add_norm = self.layernorm1(x + attn_output) # Add & Norm
		add_norm = self.dropout(add_norm)
		# feedforward
		ff_output = self.ff(attn_output) # (b, seq_len, d_model)
		add_norm2 = self.layernorm2(attn_output + ff_output) # Add & Norm
		add_norm2 = self.dropout(add_norm2)
		return add_norm2, attn_scores # (b, seq_len, d_model), (b, n_heads, seq_len, seq_len)
	

class DecoderLayer(nn.Module):
	def __init__(
			self,
			n_heads: int,
			d_model: int,
			d_k: int,
			d_v: int,
			d_ff: int,
			dropout: float
	):
		super(DecoderLayer, self).__init__()
		self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
		self.cross_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
		self.ff = PointwiseFeedforward(d_model, d_ff, dropout)
		self.layernorm1 = nn.LayerNorm(d_model)
		self.layernorm2 = nn.LayerNorm(d_model)
		self.layernorm3 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)
		
	def forward(
			self,
			x: torch.Tensor, # (b, seq_len, d_model)
			enc_output: torch.Tensor, # (b, seq_len, d_model)
			self_attn_mask: Optional[torch.Tensor] = None, # (b, seq_len, seq_len)
			cross_attn_mask: Optional[torch.Tensor] = None # (b, seq_len, seq_len)
	):
		# self-attention
		attn_output1, attn_scores1 = self.self_attn(x, x, x, self_attn_mask) # (b, seq_len, d_model)
		add_norm1 = self.layernorm1(x + attn_output1) # Add & Norm
		add_norm1 = self.dropout(add_norm1)
		# encoder-decoder attention
		attn_output2, attn_scores2 = self.cross_attn(add_norm1, enc_output, enc_output, cross_attn_mask) # (b, seq_len, d_model)
		add_norm2 = self.layernorm2(add_norm1 + attn_output2) # Add & Norm
		add_norm2 = self.dropout(add_norm2)
		# feedforward
		ff_output = self.ff(add_norm2) # (b, seq_len, d_model)
		add_norm3 = self.layernorm3(add_norm2 + ff_output) # Add & Norm
		add_norm3 = self.dropout(add_norm3)
		return add_norm3, attn_scores1, attn_scores2 # (b, seq_len, d_model), (b, n_heads, seq_len, seq_len), (b, n_heads, seq_len, seq_len)


class PositionalEncoding(nn.Module):
	def __init__(
			self,
			d_model: int,
			n_position: int
	):
		super(PositionalEncoding, self).__init__()
		# Ensure it's not a learnable parameter, just a constant
		self.register_buffer("pos_table", self._get_sinusoid_encoding_table(n_position, d_model))

	def _get_sinusoid_encoding_table(
			self,
			n_position: int,
			d_model: int
	):
		def get_position_angle_vec(position: int):
			return [position / np.power(10000, 2 * (i // 2) / d_model) for i in range(d_model)] # (d_model,)
		
		sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) # (n_position, d_model)
		sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i
		sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1

		return torch.FloatTensor(sinusoid_table).unsqueeze(0) # (1, n_position, d_model)
	
	def forward(
			self,
			x: torch.Tensor # (b, seq_len, d_model)
	):
		# clone() to ensure we don't modify the original tensor
		# detach() to ensure its not part of the computational graph
		x = x + self.pos_table[:, :x.size(1)].clone().detach()
		return x # (b, seq_len, d_model)


class Encoder(nn.Module):
	def __init__(
			self,
			src_vocab_size: int,
			d_model: int,
			d_ff: int,
			n_layers: int,
			n_heads: int,
			d_k: int,
			d_v: int,
			n_position: int,
			pad_idx: int,
			dropout: float,
			scale_emb: bool
	):
		super(Encoder, self).__init__()
		self.pad_idx = pad_idx
		self.scale_emb = scale_emb
		self.d_model = d_model
		self.embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
		self.pos_enc = PositionalEncoding(d_model, n_position)
		self.dropout = nn.Dropout(dropout)
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
		self.layers = nn.ModuleList([
			EncoderLayer(n_heads, d_model, d_k, d_v, d_ff, dropout)
			for _ in range(n_layers)
		])
		
	def forward(
			self,
			src_seq: torch.Tensor, # (b, src_len)
			src_mask: Optional[torch.Tensor] = None, # (b, src_len, src_len)
			return_attns: bool = False
	):
		attn_scores = []
		# embedding
		embs = self.embedding(src_seq) # (b, src_len, d_model)
		if self.scale_emb:
			embs *= self.d_model ** 0.5
		# positional encoding
		pos_encs = self.pos_enc(embs) # (b, src_len, d_model)
		pos_encs = self.dropout(pos_encs)
		pos_encs = self.layer_norm(pos_encs)
		# encoder layers
		layer_out = pos_encs
		for layer in self.layers:
			layer_out, layer_attn_scores = layer(layer_out, src_mask) # (b, src_len, d_model)
			if return_attns:
				attn_scores.append(layer_attn_scores)
		
		if return_attns:
			return layer_out, attn_scores # (b, src_len, d_model), [(b, n_heads, src_len, src_len)]
		return layer_out, # (b, src_len, d_model),
		

class Decoder(nn.Module):
	def __init__(
			self,
			tgt_vocab_size: int,
			d_model: int,
			d_ff: int,
			n_layers: int,
			n_heads: int,
			d_k: int,
			d_v: int,
			n_position: int,
			pad_idx: int,
			dropout: float,
			scale_emb: bool
	):
		super(Decoder, self).__init__()
		self.pad_idx = pad_idx
		self.scale_emb = scale_emb
		self.d_model = d_model
		self.embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
		self.pos_enc = PositionalEncoding(d_model, n_position)
		self.dropout = nn.Dropout(dropout)
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
		self.layers = nn.ModuleList([
			DecoderLayer(n_heads, d_model, d_k, d_v, d_ff, dropout)
			for _ in range(n_layers)
		])

	def forward(
			self,
			tgt_seq: torch.Tensor, # (b, tgt_len)
			enc_output: torch.Tensor, # (b, src_len, d_model)
			tgt_mask: Optional[torch.Tensor] = None, # (b, tgt_len, tgt_len)
			src_mask: Optional[torch.Tensor] = None, # (b, tgt_len, src_len)
			return_attns: bool = False
	):
		attn_scores1, attn_scores2 = [], []
		# embedding
		embs = self.embedding(tgt_seq) # (b, tgt_len, d_model)
		if self.scale_emb:
			embs *= self.d_model ** 0.5
		# positional encoding
		pos_encs = self.pos_enc(embs) # (b, tgt_len, d_model)
		pos_encs = self.dropout(pos_encs)
		pos_encs = self.layer_norm(pos_encs)
		# decoder layers
		layer_out = pos_encs
		for layer in self.layers:
			layer_out, layer_attn_scores1, layer_attn_scores2 = layer(layer_out, enc_output, tgt_mask, src_mask) # (b, tgt_len, d_model)
			if return_attns:
				attn_scores1.append(layer_attn_scores1)
				attn_scores2.append(layer_attn_scores2)
		
		if return_attns:
			return layer_out, attn_scores1, attn_scores2 # (b, tgt_len, d_model), [(b, n_heads, tgt_len, tgt_len)], [(b, n_heads, tgt_len, src_len)]
		return layer_out, # (b, tgt_len, d_model),
	

class Transformer:
	def __init__(
			self,
			src_vocab_size: int,
			tgt_vocab_size: int,
			d_model: int = 512,
			d_ff: int = 2048,
			n_layers: int = 6,
			n_heads: int = 8,
			d_k: int = 64,
			d_v: int = 64,
			n_position: int = 200,
			src_pad_idx: int = 0,
			tgt_pad_idx: int = 0,
			dropout: float = 0.1,
			scale_what: Literal["emb", "proj", "none"] = "proj"
	):
		assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
		assert scale_what in ["emb", "proj", "none"], "scale_what must be one of 'emb', 'proj', 'none'"
		
		self.d_model = d_model
		self.d_ff = d_ff
		self.n_layers = n_layers
		self.n_heads = n_heads
		self.d_k = d_k
		self.d_v = d_v
		self.n_position = n_position
		self.src_pad_idx = src_pad_idx
		self.tgt_pad_idx = tgt_pad_idx
		self.dropout = dropout
		self.scale_what = scale_what

		self.encoder = Encoder(
			src_vocab_size, d_model, d_ff, n_layers, n_heads, d_k, d_v,
			n_position, src_pad_idx, dropout, scale_what == "emb"
		)
		self.decoder = Decoder(
			tgt_vocab_size, d_model, d_ff, n_layers, n_heads, d_k, d_v,
			n_position, tgt_pad_idx, dropout, scale_what == "emb"
		)

		self.tgt_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	def _get_pad_mask(
			self,
			seq: torch.Tensor, # (b, seq_len)
			pad_idx: int
	):
		return (seq != pad_idx).unsqueeze(-2) # (b, 1, seq_len)
	
	def _get_lookahead_mask(
			self,
			seq: torch.Tensor # (b, seq_len)
	):
		b, seq_len = seq.size()
		return 1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).bool() # (1, seq_len, seq_len)

	def forward(
		self,
		src_seq: torch.Tensor, # (b, src_len)
		tgt_seq: torch.Tensor # (b, tgt_len)
	):
		src_mask = self._get_pad_mask(src_seq, self.src_pad_idx) # (b, 1, src_len)
		tgt_mask = self._get_pad_mask(tgt_seq, self.tgt_pad_idx) & self._get_lookahead_mask(tgt_seq) # (b, tgt_len, tgt_len)

		enc_output, *_ = self.encoder(src_seq, src_mask) # (b, src_len, d_model)
		dec_output, *_ = self.decoder(tgt_seq, enc_output, tgt_mask, src_mask) # (b, tgt_len, d_model)
		logits = self.tgt_proj(dec_output) # (b, tgt_len, tgt_vocab_size)
		if self.scale_what == "proj":
			logits *= self.d_model ** -0.5

		return logits.view(-1, logits.size(2)) # (b * tgt_len, tgt_vocab_size)
