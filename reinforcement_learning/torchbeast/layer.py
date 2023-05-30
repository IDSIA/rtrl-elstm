import torch
from torch import nn
from torch.nn import functional as F


# But the output gate is conditioned by c(t) instead of c(t-1)
class QuasiLSTMlayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, forget_bias=0.):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # weight matrices
        self.wm_z = nn.Parameter(torch.rand(hidden_dim, input_dim))
        self.wm_f = nn.Parameter(torch.rand(hidden_dim, input_dim))
        # self.wm_o = nn.Parameter(input_dim, hidden_dim)
        self.out_gate_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # weight vectors
        self.wv_z = nn.Parameter(torch.rand(1, hidden_dim))  # append B dim
        self.wv_f = nn.Parameter(torch.rand(1, hidden_dim))
        # self.wm_o = nn.Parameter(hidden_dim)

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
        out = x.view(slen * bsz, x_dim)
        out = F.linear(out, weight_matrix)
        out = out.view(slen, bsz, self.hidden_dim * 2)

        out_z, out_f = torch.split(out, (self.hidden_dim,) * 2, -1)

        output_list = []

        new_cell = hidden_prev
        # retain gradient for hidden_prev for hybrid RTRL TODO
        # for t in range(slen):
        for z_, f_ in zip(out_z, out_f):
            z_part = torch.tanh(
                z_ + self.wv_z * new_cell + self.bias_z)
            f_part = torch.sigmoid(
                f_ + self.wv_f * new_cell + self.bias_f)
            new_cell = new_cell * f_part + (1. - f_part) * z_part
            output_list.append(new_cell.clone())

        new_cells = torch.stack(output_list)  # (len, B, dim)

        out = torch.cat([x, new_cells], dim=-1)
        out = torch.sigmoid(self.out_gate_linear(out))

        out = new_cells * out

        # all outputs and the last state
        return out, new_cell.unsqueeze(0)


# Same architecture as above, but update recurrent params using RTRL;
# States are augmented with RTRL sensitivity matrices
class RTRLQuasiLSTMlayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, forget_bias=0.):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # weight matrices
        self.wm_z = nn.Parameter(
            torch.rand(hidden_dim, input_dim), requires_grad=False)
        self.wm_f = nn.Parameter(
            torch.rand(hidden_dim, input_dim), requires_grad=False)
        # self.wm_o = nn.Parameter(input_dim, hidden_dim)
        # Unlike in RTRLQuasiLSTMlayer, we move out_gate_linear outside this layer
        # self.out_gate_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # weight vectors
        self.wv_z = nn.Parameter(
            torch.rand(1, hidden_dim), requires_grad=False)  # append B dim
        self.wv_f = nn.Parameter(
            torch.rand(1, hidden_dim), requires_grad=False)
        # self.wm_o = nn.Parameter(hidden_dim)

        # biases
        self.bias_z = nn.Parameter(
            torch.rand(1, hidden_dim), requires_grad=False)
        self.bias_f = nn.Parameter(
            torch.rand(1, hidden_dim), requires_grad=False)
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

    def forward(self, x, state=None, is_actor=True):
        if is_actor:
            return self.forward_actor(x, state)
        else:
            return self.forward_learner(x, state)

    # the basic forward to be used by actors.
    # carry both rnn_state and RTRL states
    def forward_actor(self, x, state=None):
        # Important: this function is called step-by-step together with the logic
        # that checks for the end of an episode to reset state.

        # is_for_training is False for actors
        # x shape: (len, B, n_head * d_head)
        slen, bsz, x_dim = x.size()

        assert state is not None
        # state is a tuple (containing RNN state, (tuple with one RTRL state for each parameters))
        hidden_prev, rtrl_state = state
        hidden_prev = hidden_prev.squeeze(0)  # layer dim compat.
        Z_state, F_state, wz_state, wf_state, bz_state, bf_state = rtrl_state

        # removing layer dim, non priority but better to remove this
        Z_state = Z_state.squeeze(0)
        F_state = F_state.squeeze(0)
        wz_state = wz_state.squeeze(0)
        wf_state = wf_state.squeeze(0)
        bz_state = bz_state.squeeze(0)
        bf_state = bf_state.squeeze(0)

        weight_matrix = torch.cat([self.wm_z, self.wm_f], dim=0)
        out = x.view(slen * bsz, x_dim)
        out = F.linear(out, weight_matrix)
        out = out.view(slen, bsz, self.hidden_dim * 2)

        out_z, out_f = torch.split(out, (self.hidden_dim,) * 2, -1)

        output_list = []
        new_cell = hidden_prev
        # retain gradient for hidden_prev for hybrid RTRL TODO
        # for t in range(slen):
        for x_, pre_z_, pre_f_ in zip(x, out_z, out_f):
            z_part = torch.tanh(
                pre_z_ + self.wv_z * new_cell + self.bias_z)
            f_part = torch.sigmoid(
                pre_f_ + self.wv_f * new_cell + self.bias_f)

            # RTRL states update
            with torch.no_grad():
                # tanh on z, sigmoid on f
                zf_tmp = (1. - f_part) * (1. - z_part ** 2)
                fz_tmp = (new_cell - z_part) * (1. - f_part) * f_part
                common_tmp = f_part + zf_tmp * self.wv_z + fz_tmp * self.wv_f

                Z_state = common_tmp.unsqueeze(-1) * Z_state + torch.bmm(
                    zf_tmp.unsqueeze(-1), x_.unsqueeze(1))  # broadcast mul

                F_state = common_tmp.unsqueeze(-1) * F_state + torch.bmm(
                    fz_tmp.unsqueeze(-1), x_.unsqueeze(1))

                # recurrent scalers
                wz_state = new_cell * zf_tmp + common_tmp * wz_state
                wf_state = new_cell * fz_tmp + common_tmp * wf_state

                # biases
                bz_state = zf_tmp + common_tmp * bz_state
                bf_state = fz_tmp + common_tmp * bf_state

            # update state
            new_cell = new_cell * f_part + (1. - f_part) * z_part
            output_list.append(new_cell.clone())

        new_cells = torch.stack(output_list)  # (len, B, dim)

        # out = torch.cat([x, new_cells], dim=-1)
        # out = torch.sigmoid(self.out_gate_linear(out))

        # out = new_cells * out

        rtrl_states = (
            Z_state.clone().unsqueeze(0), F_state.clone().unsqueeze(0),
            wz_state.clone().unsqueeze(0), wf_state.clone().unsqueeze(0),
            bz_state.clone().unsqueeze(0), bf_state.clone().unsqueeze(0))

        return new_cells, (new_cell.unsqueeze(0), rtrl_states)

    # forward function to be called by the 'learner' to train the model.
    # we then call: total_loss.backward(), forward_rtrl(), and finally optimizer.step()
    # We need:
    # - cell states c_t for all time steps, including the first hidden state
    #   Above is needed for 2 reasons:
    #   (1) to get top gradients (see retain_grad below)
    #   (2) c(t-1) is needed for each step of RTRL
    # - all 'z' and 'f' outputs
    # learner forward does not need to do RTRL inside; as it is immediately followed by `compute_grad_rtrl`
    def forward_learner(self, x, state=None):
        # Important: this function is called step-by-step together with the logic
        # that checks for the end of an episode to reset state.

        # x shape: (len, B, n_head * d_head)
        # state is a tuple (containing RNN state, (tuple with one RTRL state for each parameters))
        slen, bsz, x_dim = x.size()

        # 1. Regular forward pass ======
        assert state is not None
        # we only need the RNN state for this forward pass
        hidden_prev = state[0].squeeze(0)  # layer dim compat.

        weight_matrix = torch.cat([self.wm_z, self.wm_f], dim=0)
        out = x.view(slen * bsz, x_dim)
        out = F.linear(out, weight_matrix)
        out = out.view(slen, bsz, self.hidden_dim * 2)

        out_z, out_f = torch.split(out, (self.hidden_dim,) * 2, -1)

        # output_list = [hidden_prev.clone()]  # this is for training.
        output_list = []
        z_list = []
        f_list = []

        new_cell = hidden_prev
        # retain gradient for hidden_prev for hybrid RTRL TODO
        # for t in range(slen):
        for z_, f_ in zip(out_z, out_f):
            z_part = torch.tanh(
                z_ + self.wv_z * new_cell + self.bias_z)
            f_part = torch.sigmoid(
                f_ + self.wv_f * new_cell + self.bias_f)
            new_cell = new_cell * f_part + (1. - f_part) * z_part
            output_list.append(new_cell.clone())

            z_list.append(z_part.clone().detach())
            f_list.append(f_part.clone().detach())

        new_cells = torch.stack(output_list)  # (len, B, dim)
        new_cells.requires_grad_()
        new_cells.retain_grad()

        z_out = torch.stack(z_list)
        f_out = torch.stack(f_list)

        # return out, z_out, f_out, new_cells[-1].unsqueeze(0)
        return new_cells, z_out, f_out, new_cells[-1].unsqueeze(0)

    @torch.no_grad()
    def rtrl_zero_grad(self):
        self.wm_z.grad = torch.zeros_like(self.wm_z)
        self.wm_f.grad = torch.zeros_like(self.wm_f)

        self.wv_z.grad = torch.zeros_like(self.wv_z)
        self.wv_f.grad= torch.zeros_like(self.wv_f)

        self.bias_z.grad = torch.zeros_like(self.bias_z)
        self.bias_f.grad = torch.zeros_like(self.bias_f)

    # We assume that loss.backward() is called before, and thus, top_gradients are available.
    @torch.no_grad()
    def compute_grad_rtrl(self, x, rtrl_states, top_gradients, rnn_state_tm1, z_out, f_out):
        # slen, _, _ = x.size()
        # assert state is not None
        # we need both RNN and RTRL states
        # assert state is not None
        assert top_gradients is not None
        # rnn_state_tm1, z_out, f_out, rtrl_states = state
        Z_state, F_state, wz_state, wf_state, bz_state, bf_state = rtrl_states

        # update all RTRL states, but also compute gradients already; we only carry over the last RTRL state to the next segment
        for x_, top_grad_, c_tm1_, z_, f_ in zip(x, top_gradients, rnn_state_tm1, z_out, f_out):
            # tanh on z, sigmoid on f
            zf_tmp = (1. - f_) * (1. - z_ ** 2)
            fz_tmp = (c_tm1_ - z_) * (1. - f_) * f_
            common_tmp = f_ + zf_tmp * self.wv_z + fz_tmp * self.wv_f

            Z_state = common_tmp.unsqueeze(-1) * Z_state + torch.bmm(
                zf_tmp.unsqueeze(-1), x_.unsqueeze(1))  # broadcast mul

            F_state = common_tmp.unsqueeze(-1) * F_state + torch.bmm(
                fz_tmp.unsqueeze(-1), x_.unsqueeze(1))

            # recurrent scalers
            wz_state = c_tm1_ * zf_tmp + common_tmp * wz_state
            wf_state = c_tm1_ * fz_tmp + common_tmp * wf_state

            # biases
            bz_state = zf_tmp + common_tmp * bz_state
            bf_state = fz_tmp + common_tmp * bf_state

            # compute gradients, sum over batch dim
            self.wm_z.grad += (top_grad_.unsqueeze(-1) * Z_state).sum(dim=0)
            self.wm_f.grad += (top_grad_.unsqueeze(-1) * F_state).sum(dim=0)

            self.wv_z.grad += (top_grad_ * wz_state).sum(dim=0)
            self.wv_f.grad += (top_grad_ * wf_state).sum(dim=0)

            self.bias_z.grad += (top_grad_ * bz_state).sum(dim=0)
            self.bias_f.grad += (top_grad_ * bf_state).sum(dim=0)

        rtrl_state = (Z_state.clone(), F_state.clone(), wz_state.clone(), wf_state.clone(), bz_state.clone(), bf_state.clone())

        return rtrl_state
