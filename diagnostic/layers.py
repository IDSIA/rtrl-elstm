import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def elu_p1(x):
    return F.elu(x, 1., False) + 1.


@torch.jit.script
def sum_norm(x):
    return x / x.sum(-1, keepdim=True)


# Quasi RNN-like https://arxiv.org/abs/1611.01576
# But the output gate is conditioned by c(t) instead of c(t-1)
class QuasiLSTMlayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, forget_bias=0.):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # weight matrices
        self.wm_z = nn.Parameter(torch.rand(hidden_dim, input_dim))
        self.wm_f = nn.Parameter(torch.rand(hidden_dim, input_dim))

        # weight vectors
        self.wv_z = nn.Parameter(torch.rand(1, hidden_dim))  # append B dim
        self.wv_f = nn.Parameter(torch.rand(1, hidden_dim))

        # biases
        self.bias_z = nn.Parameter(torch.rand(1, hidden_dim))
        self.bias_f = nn.Parameter(torch.rand(1, hidden_dim))
        self.forget_bias = forget_bias

        self.init_weights()

    def init_weights(self):
        torch.nn.init.normal_(self.wm_z, mean=0.0, std=0.1)
        torch.nn.init.normal_(self.wm_f, mean=0.0, std=0.1)

        torch.nn.init.normal_(self.wv_z, mean=0.0, std=0.1)
        torch.nn.init.normal_(self.wv_f, mean=0.0, std=0.1)

        torch.nn.init.normal_(self.bias_z, mean=0.0, std=0.1)
        torch.nn.init.normal_(self.bias_f, mean=0.0, std=0.1)
        with torch.no_grad():
            self.bias_f.copy_(self.bias_f + self.forget_bias)

    def forward(self, x, state=None):
        # x shape: (len, B, n_head * d_head)
        # state is a tuple
        slen, bsz, x_dim = x.size()

        if state is None:
            hidden_prev = torch.zeros(
                [bsz, self.hidden_dim], device=x.device)
        else:
            hidden_prev = state.squeeze(0)  # layer dim compat.

        weight_matrix = torch.cat([self.wm_z, self.wm_f], dim=0)
        out = x.reshape(slen * bsz, x_dim)
        out = F.linear(out, weight_matrix)
        out = out.view(slen, bsz, self.hidden_dim * 2)

        out_z, out_f = torch.split(out, (self.hidden_dim,) * 2, -1)

        output_list = []

        new_cell = hidden_prev
        for z_, f_ in zip(out_z, out_f):
            z_part = torch.tanh(
                z_ + self.wv_z * new_cell + self.bias_z)
            f_part = torch.sigmoid(
                f_ + self.wv_f * new_cell + self.bias_f)
            new_cell = new_cell * f_part + (1. - f_part) * z_part
            output_list.append(new_cell.clone())

        new_cells = torch.stack(output_list)  # (len, B, dim)

        return new_cells
