import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def elu_p1(x):
    return F.elu(x, 1., False) + 1.


@torch.jit.script
def sum_norm(x):
    return x / x.sum(-1, keepdim=True)


class RTRLQuasiLSTMlayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, forget_bias=0.):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # weight matrices
        self.wm_z = nn.Parameter(torch.rand(hidden_dim, input_dim), requires_grad=False)
        self.wm_f = nn.Parameter(torch.rand(hidden_dim, input_dim), requires_grad=False)

        # weight vectors
        self.wv_z = nn.Parameter(torch.rand(1, hidden_dim), requires_grad=False)  # append B dim
        self.wv_f = nn.Parameter(torch.rand(1, hidden_dim), requires_grad=False)

        # biases
        self.bias_z = nn.Parameter(torch.rand(1, hidden_dim), requires_grad=False)
        self.bias_f = nn.Parameter(torch.rand(1, hidden_dim), requires_grad=False)
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

    # RNN and RTRL states
    def get_init_states(self, batch_size, device):
        rnn_state = torch.zeros(
            [batch_size, self.hidden_dim], device=device)

        Z_state = torch.zeros(
            [batch_size, self.hidden_dim, self.input_dim], device=device)
        F_state = torch.zeros(
            [batch_size, self.hidden_dim, self.input_dim], device=device)

        wz_state = torch.zeros(
            [batch_size, self.hidden_dim], device=device)
        wf_state = torch.zeros(
            [batch_size, self.hidden_dim], device=device)

        bz_state = torch.zeros(
            [batch_size, self.hidden_dim], device=device)
        bf_state = torch.zeros(
            [batch_size, self.hidden_dim], device=device)

        rtrl_states = Z_state, F_state, wz_state, wf_state, bz_state, bf_state

        return (rnn_state, rtrl_states)

    # `Cell' like step by step process
    def forward(self, x, state=None):
        # x shape: (B, dim)
        assert state is not None
        hidden_prev, rtrl_states = state
        Z_state, F_state, wz_state, wf_state, bz_state, bf_state = rtrl_states

        weight_matrix = torch.cat([self.wm_z, self.wm_f], dim=0)
        out = F.linear(x, weight_matrix)

        out_z, out_f = torch.split(out, (self.hidden_dim,) * 2, -1)

        new_cell = hidden_prev

        z_part = torch.tanh(
            out_z + self.wv_z * new_cell + self.bias_z)
        f_part = torch.sigmoid(
            out_f + self.wv_f * new_cell + self.bias_f)
        new_cell = new_cell * f_part + (1. - f_part) * z_part

        # update RTRL states
        with torch.no_grad():
            # tanh on z, sigmoid on f
            zf_tmp = (1. - f_part) * (1. - z_part ** 2)
            fz_tmp = (hidden_prev - z_part) * (1. - f_part) * f_part
            common_tmp = f_part + zf_tmp * self.wv_z + fz_tmp * self.wv_f

            Z_state = common_tmp.unsqueeze(-1) * Z_state + torch.bmm(
                zf_tmp.unsqueeze(-1), x.unsqueeze(1))  # broadcast mul

            F_state = common_tmp.unsqueeze(-1) * F_state + torch.bmm(
                fz_tmp.unsqueeze(-1), x.unsqueeze(1))

            # recurrent scalers
            wz_state = hidden_prev * zf_tmp + common_tmp * wz_state
            wf_state = hidden_prev * fz_tmp + common_tmp * wf_state

            # biases
            bz_state = zf_tmp + common_tmp * bz_state
            bf_state = fz_tmp + common_tmp * bf_state

        rtrl_states = (
            Z_state, F_state, wz_state, wf_state, bz_state, bf_state)

        state = (new_cell.clone(), rtrl_states)

        # all outputs
        return new_cell.clone(), state


if __name__ == '__main__':
    # Gradient test for RTRL for Quasi-LSTM
    # import torch
    torch.manual_seed(123)
    torch.set_default_tensor_type(torch.DoubleTensor)

    rel_threshold = 1e-4

    # from https://github.com/idiap/fast-transformers/blob/master/tests/causal_product/test_causal_product_gpu.py
    def max_relative_error(a, b, eps=1e-6):
        return torch.abs((b - a) / (torch.abs(b) + eps)).max().item()

    print('##########################')
    print('# Test RTRL')
    print('##########################')

    #################################

    input_dim = 3  # 3
    hidden_dim = 5  # 5
    slen = 70  # 70
    bsz = 10  # 2
    print(f"slen {slen} bsz {bsz} hidden_dim {hidden_dim} input_dim {input_dim}")

    wm_z0 = torch.rand(hidden_dim, input_dim, device='cuda')
    wm_f0 = torch.rand(hidden_dim, input_dim, device='cuda')

    wv_z0 = torch.rand(1, hidden_dim, device='cuda')
    wv_f0 = torch.rand(1, hidden_dim, device='cuda')

    # wv_z0 = torch.zeros(1, hidden_dim, device='cuda')
    # wv_f0 = torch.zeros(1, hidden_dim, device='cuda')

    bias_z0 = torch.rand(1, hidden_dim, device='cuda')
    bias_f0 = torch.rand(1, hidden_dim, device='cuda')

    #################################

    wm_z1 = torch.zeros(hidden_dim, input_dim, requires_grad=True, device='cuda')
    wm_f1 = torch.zeros(hidden_dim, input_dim, requires_grad=True, device='cuda')

    wv_z1 = torch.zeros(1, hidden_dim, requires_grad=True, device='cuda')  # append B dim
    wv_f1 = torch.zeros(1, hidden_dim, requires_grad=True, device='cuda')

    bias_z1 = torch.zeros(1, hidden_dim, requires_grad=True, device='cuda')
    bias_f1 = torch.zeros(1, hidden_dim, requires_grad=True, device='cuda')

    wm_z1.data = wm_z0.data
    wm_f1.data = wm_f0.data

    wv_z1.data = wv_z0.data
    wv_f1.data = wv_f0.data

    bias_z1.data = bias_z0.data
    bias_f1.data = bias_f0.data

    #################################
    # Common to both passes

    input_x = torch.rand([slen, bsz, input_dim], device='cuda')
    init_state = torch.zeros([bsz, hidden_dim], device='cuda') + .5

    #################################

    weight_matrix1 = torch.cat([wm_z1, wm_f1], dim=0)
    out1 = input_x.reshape(slen * bsz, input_dim)
    out1 = F.linear(out1, weight_matrix1)
    out1 = out1.view(slen, bsz, hidden_dim * 2)

    out_z1, out_f1 = torch.split(out1, (hidden_dim,) * 2, -1)
    new_cell1 = init_state

    output_list1 = []
    rnn_state_tm1_list = [init_state.clone()]
    f_out_list = []
    z_out_list = []

    # forward and compute gradients using PyTorch AD
    for t in range(slen):
        z_part1 = torch.tanh(
            out_z1[t] + wv_z1 * new_cell1 + bias_z1)
        f_part1 = torch.sigmoid(
            out_f1[t] + wv_f1 * new_cell1 + bias_f1)
        new_cell1 = new_cell1 * f_part1 + (1. - f_part1) * z_part1
        output_list1.append(new_cell1.clone())
        f_out_list.append(f_part1.clone())
        z_out_list.append(z_part1.clone())
        if t < slen - 1:
            rnn_state_tm1_list.append(new_cell1.clone())

    rnn_state_tm1 = torch.stack(rnn_state_tm1_list)
    f_out = torch.stack(f_out_list)
    z_out = torch.stack(z_out_list)

    all_outs1 = torch.stack(output_list1)
    all_outs1.retain_grad()

    # grad
    loss1 = all_outs1.sum() * 10
    wm_z1.retain_grad()
    wm_f1.retain_grad()

    wv_z1.retain_grad()
    wv_f1.retain_grad()

    bias_z1.retain_grad()
    bias_f1.retain_grad()

    loss1.backward()

    ###########################################################################

    top_gradients = all_outs1.grad

    # RTRL states
    Z_state = torch.zeros([bsz, hidden_dim, input_dim], device='cuda')
    F_state = torch.zeros([bsz, hidden_dim, input_dim], device='cuda')

    wz_state = torch.zeros([bsz, hidden_dim], device='cuda')
    wf_state = torch.zeros([bsz, hidden_dim], device='cuda')

    bz_state = torch.zeros([bsz, hidden_dim], device='cuda')
    bf_state = torch.zeros([bsz, hidden_dim], device='cuda')

    wm_z2_grad = []
    wm_f2_grad = []

    wv_z2_grad = []
    wv_f2_grad= []

    bias_z2_grad = []
    bias_f2_grad = []

    with torch.no_grad():
        for x_, top_grad, c_tm1, z_, f_ in zip(input_x, top_gradients, rnn_state_tm1, z_out, f_out):
            # tanh on z, sigmoid on f
            zf_tmp = (1. - f_) * (1. - z_ ** 2)
            fz_tmp = (c_tm1 - z_) * (1. - f_) * f_

            common_tmp = f_ + zf_tmp * wv_z1 + fz_tmp * wv_f1

            Z_state = common_tmp.unsqueeze(-1) * Z_state + torch.bmm(
                zf_tmp.unsqueeze(-1), x_.unsqueeze(1))  # broadcast mul

            F_state = common_tmp.unsqueeze(-1) * F_state + torch.bmm(
                fz_tmp.unsqueeze(-1), x_.unsqueeze(1))

            # recurrent scalers
            wz_state = c_tm1 * zf_tmp + common_tmp * wz_state
            wf_state = c_tm1 * fz_tmp + common_tmp * wf_state

            # biases
            bz_state = zf_tmp + common_tmp * bz_state
            bf_state = fz_tmp + common_tmp * bf_state

            # compute gradients
            wm_z2_grad.append(top_grad.unsqueeze(-1) * Z_state.clone())
            wm_f2_grad.append(top_grad.unsqueeze(-1) * F_state.clone())

            wv_z2_grad.append(top_grad * wz_state.clone())
            wv_f2_grad.append(top_grad * wf_state.clone())

            bias_z2_grad.append(top_grad * bz_state.clone())
            bias_f2_grad.append(top_grad * bf_state.clone())

    wm_z2_grad = torch.stack(wm_z2_grad).view(slen * bsz, hidden_dim, input_dim).sum(0)
    wm_f2_grad = torch.stack(wm_f2_grad).view(slen * bsz, hidden_dim, input_dim).sum(0)

    wv_z2_grad = torch.stack(wv_z2_grad).view(slen * bsz, hidden_dim).sum(0)
    wv_f2_grad = torch.stack(wv_f2_grad).view(slen * bsz, hidden_dim).sum(0)

    bias_z2_grad = torch.stack(bias_z2_grad).view(slen * bsz, hidden_dim).sum(0)
    bias_f2_grad = torch.stack(bias_f2_grad).view(slen * bsz, hidden_dim).sum(0)

    ###########################################################################

    print("======== Z Weight matrix =================================")
    for h in range(hidden_dim):
        for i in range(input_dim):
            print(f"h={h}, i={i}")
            print(f"wm_z grad BPTT: {wm_z1.grad[h][i]}")
            print(f"wm_z grad RTRL: {wm_z2_grad[h][i]}")
            rel_error = max_relative_error(wm_z1.grad[h][i], wm_z2_grad[h][i])
            assert rel_error < rel_threshold, f"{rel_error} >= {rel_threshold}"
            print("pass!")
            print("---------------------------------------")

    print("========= F Weight matrix ================================")
    for h in range(hidden_dim):
        for i in range(input_dim):
            print(f"h={h}, i={i}")
            print(f"wm_f grad BPTT: {wm_f1.grad[h][i]}")
            print(f"wm_f grad RTRL: {wm_f2_grad[h][i]}")
            rel_error = max_relative_error(wm_f1.grad[h][i], wm_f2_grad[h][i])
            assert rel_error < rel_threshold, f"{rel_error} >= {rel_threshold}"
            print("pass!")
            print("---------------------------------------")

    print(" Weight vectors ---------------------------------------")
    for h in range(hidden_dim):
        print(f"wv_z grad BPTT: {wv_z1.grad[0][h]}")
        print(f"wv_z grad RTRL: {wv_z2_grad[h]}")
        rel_error = max_relative_error(wv_z1.grad[0][h], wv_z2_grad[h])
        assert rel_error < rel_threshold, f"{rel_error} >= {rel_threshold}"
        print("pass!")
        print("---------------------------------------")

    for h in range(hidden_dim):
        print(f"wv_f grad BPTT: {wv_f1.grad[0][h]}")
        print(f"wv_f grad RTRL: {wv_f2_grad[h]}")
        rel_error = max_relative_error(wv_f1.grad[0][h], wv_f2_grad[h])
        assert rel_error < rel_threshold, f"{rel_error} >= {rel_threshold}"
        print("pass!")
        print("---------------------------------------")

    print("===  Biases =================================")
    for h in range(hidden_dim):
        print(f"bias_z grad BPTT: {bias_z1.grad[0][h]}")
        print(f"bias_z grad RTRL: {bias_z2_grad[h]}")
        rel_error = max_relative_error(bias_z1.grad[0][h], bias_z2_grad[h])
        assert rel_error < rel_threshold, f"{rel_error} >= {rel_threshold}"
        print("pass!")
        print("---------------------------------------")

    for h in range(hidden_dim):
        print(f"bias_f grad BPTT: {bias_f1.grad[0][h]}")
        print(f"bias_f grad RTRL: {bias_f2_grad[h]}")
        rel_error = max_relative_error(bias_f1.grad[0][h], bias_f2_grad[h])
        assert rel_error < rel_threshold, f"{rel_error} >= {rel_threshold}"
        print("pass!")
        print("---------------------------------------")

    print("All tests pass.")
