# Contains model implementations.

import torch
import torch.nn as nn


from layers import QuasiLSTMlayer
from rtrl_layers import RTRLQuasiLSTMlayer


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    # return number of parameters
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_grad(self):
        # More efficient than optimizer.zero_grad() according to:
        # Szymon Migacz "PYTORCH PERFORMANCE TUNING GUIDE" at GTC-21.
        # - doesn't execute memset for every parameter
        # - memory is zeroed-out by the allocator in a more efficient way
        # - backward pass updates gradients with "=" operator (write) (unlike
        # zero_grad() which would result in "+=").
        # In PyT >= 1.7, one can do `model.zero_grad(set_to_none=True)`
        for p in self.parameters():
            p.grad = None

    def print_params(self):
        for p in self.named_parameters():
            print(p)


# Pure PyTorch LSTM model
class LSTMModel(BaseModel):
    def __init__(self, emb_dim, hidden_size, in_vocab_size, out_vocab_size,
                 dropout=0.0, num_layers=1, no_embedding=False):
        super().__init__()
        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.hidden_size = hidden_size

        self.no_embedding = no_embedding
        rnn_input_size = in_vocab_size

        if not no_embedding:
            self.embedding = nn.Embedding(
                num_embeddings=in_vocab_size, embedding_dim=emb_dim)
            rnn_input_size = emb_dim
        else:
            self.num_classes = in_vocab_size

        self.rnn_func = nn.LSTM(
            input_size=rnn_input_size, hidden_size=hidden_size,
            num_layers=num_layers)

        self.dropout = dropout
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(hidden_size, out_vocab_size)

    def forward(self, x):
        if self.no_embedding:
            out = torch.nn.functional.one_hot(x, self.num_classes).permute(1, 0, 2).float()
        else:
            out = self.embedding(x).permute(1, 0, 2)  # seq dim first

        # if self.dropout:
        #     out = self.dropout(out)
        out, _ = self.rnn_func(out)

        if self.dropout:
            out = self.dropout(out)
        logits = self.out_layer(out).permute(1, 0, 2)

        return logits


class QuasiLSTMModel(BaseModel):
    def __init__(self, emb_dim, hidden_size, in_vocab_size, out_vocab_size,
                 dropout=0.0, num_layers=1, no_embedding=False):
        super().__init__()
        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.hidden_size = hidden_size

        self.no_embedding = no_embedding
        rnn_input_size = in_vocab_size
        if not no_embedding:
            self.embedding = nn.Embedding(
                num_embeddings=in_vocab_size, embedding_dim=emb_dim)
            rnn_input_size = emb_dim
        else:
            self.num_classes = in_vocab_size

        self.rnn_func = QuasiLSTMlayer(
            input_dim=rnn_input_size, hidden_dim=hidden_size)

        self.output_gate = nn.Linear(
            rnn_input_size + hidden_size, hidden_size)

        self.dropout = dropout
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(hidden_size, out_vocab_size)

    def forward(self, x):
        if self.no_embedding:
            out = torch.nn.functional.one_hot(x, self.num_classes).permute(1, 0, 2).float()
        else:
            out = self.embedding(x).permute(1, 0, 2)  # seq dim first

        # if self.dropout:
        #     out = self.dropout(out)
        cell_out = self.rnn_func(out)

        gate_out = self.output_gate(torch.cat([out, cell_out], dim=-1))
        gate_out = torch.sigmoid(gate_out)
        gate_out = cell_out * gate_out

        if self.dropout:
            gate_out = self.dropout(gate_out)
        logits = self.out_layer(gate_out).permute(1, 0, 2)

        return logits


class RTRLQuasiLSTMModel(BaseModel):
    def __init__(self, emb_dim, hidden_size, in_vocab_size, out_vocab_size,
                 dropout=0.0, num_layers=1, no_embedding=False):
        super().__init__()
        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.hidden_size = hidden_size

        self.no_embedding = no_embedding
        rnn_input_size = in_vocab_size
        self.num_classes = in_vocab_size
        if not no_embedding:
            self.embedding = nn.Embedding(
                num_embeddings=in_vocab_size, embedding_dim=emb_dim)
            rnn_input_size = emb_dim

        self.rnn_func = RTRLQuasiLSTMlayer(
            input_dim=rnn_input_size, hidden_dim=hidden_size)

        self.output_gate = nn.Linear(
            rnn_input_size + hidden_size, hidden_size)

        self.dropout = dropout
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(hidden_size, out_vocab_size)

    def forward(self, x, state):
        if self.no_embedding:
            out = torch.nn.functional.one_hot(x, self.num_classes).float()
        else:
            out = self.embedding(x)  # seq dim first

        # RTRLQuasiLSTMlayer can take inputs of shape (B, dim)
        cell_out, state = self.rnn_func(out, state)
        cell_out.requires_grad_()
        cell_out.retain_grad()

        gate_out = self.output_gate(torch.cat([out, cell_out], dim=-1))
        gate_out = torch.sigmoid(gate_out)
        gate_out = cell_out * gate_out

        logits = self.out_layer(gate_out)

        return logits, cell_out, state

    def compute_gradient_rtrl(self, top_grad_, rtrl_state):
        Z_state, F_state, wz_state, wf_state, bz_state, bf_state = rtrl_state

        self.rnn_func.wm_z.grad += (top_grad_.unsqueeze(-1) * Z_state).sum(dim=0)
        self.rnn_func.wm_f.grad += (top_grad_.unsqueeze(-1) * F_state).sum(dim=0)

        self.rnn_func.wv_z.grad += (top_grad_ * wz_state).sum(dim=0)
        self.rnn_func.wv_f.grad += (top_grad_ * wf_state).sum(dim=0)

        self.rnn_func.bias_z.grad += (top_grad_ * bz_state).sum(dim=0)
        self.rnn_func.bias_f.grad += (top_grad_ * bf_state).sum(dim=0)

    def get_init_states(self, batch_size, device):
        return self.rnn_func.get_init_states(batch_size, device)

    def rtrl_reset_grad(self):
        self.rnn_func.wm_z.grad = torch.zeros_like(self.rnn_func.wm_z)
        self.rnn_func.wm_f.grad = torch.zeros_like(self.rnn_func.wm_f)

        self.rnn_func.wv_z.grad = torch.zeros_like(self.rnn_func.wv_z)
        self.rnn_func.wv_f.grad = torch.zeros_like(self.rnn_func.wv_f)

        self.rnn_func.bias_z.grad = torch.zeros_like(self.rnn_func.bias_z)
        self.rnn_func.bias_f.grad = torch.zeros_like(self.rnn_func.bias_f)
