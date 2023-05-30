import nest

import torch
from torch import nn
from torch.nn import functional as F

from torchbeast.layer import QuasiLSTMlayer, RTRLQuasiLSTMlayer


# Baseline model from torchbeast
class Net(nn.Module):
    def __init__(self, num_actions, conv_scale=1, use_lstm=False,
                 hidden_size=256, freeze_conv=False, freeze_fc=False):
        super(Net, self).__init__()
        self.num_actions = num_actions
        self.use_lstm = use_lstm
        self.hidden_size = hidden_size

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = 3
        base_num_ch = [16, 32, 32]
        scaled_num_ch = [c * conv_scale for c in base_num_ch]
        for num_ch in scaled_num_ch:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(2048 * conv_scale, 256)

        if freeze_conv:
            # freeze all vision params
            for param in self.feat_convs.parameters():
                param.requires_grad = False
            for param in self.resnet1.parameters():
                param.requires_grad = False
            for param in self.resnet2.parameters():
                param.requires_grad = False
        if freeze_fc:
            for param in self.fc.parameters():
                param.requires_grad = False

        # FC output size + last reward.
        core_output_size = self.fc.out_features + 1

        if use_lstm:
            self.core = nn.LSTM(core_output_size, hidden_size, num_layers=1)
            core_output_size = hidden_size

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state):
        x = inputs["frame"]

        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = nest.map(nd.mul, core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (action, policy_logits, baseline), core_state


class QuasiLSTMNet(nn.Module):
    def __init__(self, num_actions, conv_scale=1, hidden_size=256,
                 freeze_conv=False, freeze_fc=False):
        super().__init__()

        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.rnn_num_layers = 1  # hard coded for RTRL

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = 3
        base_num_ch = [16, 32, 32]
        scaled_num_ch = [c * conv_scale for c in base_num_ch]
        for num_ch in scaled_num_ch:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(2048 * conv_scale, hidden_size)

        if freeze_conv:
            # freeze all vision params
            for param in self.feat_convs.parameters():
                param.requires_grad = False
            for param in self.resnet1.parameters():
                param.requires_grad = False
            for param in self.resnet2.parameters():
                param.requires_grad = False
        if freeze_fc:
            for param in self.fc.parameters():
                param.requires_grad = False

        # FC output size + last reward.
        core_output_size = self.fc.out_features + 1

        self.core = QuasiLSTMlayer(core_output_size, hidden_size)
        core_output_size = hidden_size

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        return tuple(
            torch.zeros(self.rnn_num_layers, batch_size, self.hidden_size)
            for _ in range(1)
        )

    def forward(self, inputs, core_state):
        x = inputs["frame"]

        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)

        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs["done"]).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            # Reset core state to zero whenever an episode ended.
            # Make `done` broadcastable with (num_layers, B, hidden_size)
            # states:
            nd = nd.view(1, -1, 1)
            core_state = nest.map(nd.mul, core_state)
            core_state = core_state[0]  # untuple
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_state = (core_state,)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (action, policy_logits, baseline), core_state


class RTRLQuasiLSTMNet(nn.Module):
    def __init__(self, num_actions, conv_scale=1, hidden_size=256,
                 freeze_conv=False, freeze_fc=False):
        super().__init__()

        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.rnn_num_layers = 1  # hard coded for RTRL

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = 3
        base_num_ch = [16, 32, 32]
        scaled_num_ch = [c * conv_scale for c in base_num_ch]
        for num_ch in scaled_num_ch:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(2048 * conv_scale, hidden_size)

        if freeze_conv:
            # freeze all vision params
            for param in self.feat_convs.parameters():
                param.requires_grad = False
            for param in self.resnet1.parameters():
                param.requires_grad = False
            for param in self.resnet2.parameters():
                param.requires_grad = False
            for param in self.fc.parameters():
                param.requires_grad = False
        if freeze_fc:
            for param in self.fc.parameters():
                param.requires_grad = False

        # FC output size + last reward.
        self.rnn_input_dim = self.fc.out_features + 1

        self.core = RTRLQuasiLSTMlayer(self.rnn_input_dim, hidden_size)
        core_output_size = hidden_size

        self.output_gate = nn.Linear(self.rnn_input_dim + hidden_size, hidden_size)

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        return (
            torch.zeros(1, batch_size, self.hidden_size),
            (torch.zeros(1, batch_size, self.hidden_size, self.rnn_input_dim),
             torch.zeros(1, batch_size, self.hidden_size, self.rnn_input_dim),
             torch.zeros(1, batch_size, self.hidden_size),
             torch.zeros(1, batch_size, self.hidden_size),
             torch.zeros(1, batch_size, self.hidden_size),
             torch.zeros(1, batch_size, self.hidden_size),)
        )

    def compute_grad_rtrl(self, cached_input, notdone, rnn_outs_tm1, z_outs, f_outs, core_state, top_gradient):
        for input, nd, top_grad, rnn_tm1_, z_, f_ in zip(cached_input.unbind(), notdone.unbind(), top_gradient.unbind(), rnn_outs_tm1.unbind(), z_outs.unbind(), f_outs.unbind()):
            # Reset core state to zero whenever an episode ended.
            # Make `done` broadcastable with (num_layers, B, hidden_size)
            # learner's `core_state` only contains the cell state
            Z_state, F_state, wz_state, wf_state, bz_state, bf_state = core_state
            nd = nd.view(-1, 1)
            wz_state = nest.map(nd.mul, wz_state)
            wf_state = nest.map(nd.mul, wf_state)
            bz_state = nest.map(nd.mul, bz_state)
            bf_state = nest.map(nd.mul, bf_state)

            nd = nd.view(-1, 1, 1)
            Z_state = nest.map(nd.mul, Z_state)
            F_state = nest.map(nd.mul, F_state)

            core_state = (Z_state, F_state, wz_state, wf_state, bz_state, bf_state)
            core_state = self.core.compute_grad_rtrl(input.unsqueeze(0), core_state, top_grad.unsqueeze(0), rnn_tm1_.unsqueeze(0), z_.unsqueeze(0), f_.unsqueeze(0))

    # this is the generic forward, input/output state is consistently the last (rnn_state, rtrl_state)
    def forward_learner(self, inputs, core_state):
        x = inputs["frame"]

        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)

        core_input = core_input.view(T, B, -1)
        core_output_list = []
        # learner also need to store zf output to avoid recompution in the RTRL pass
        z_output_list = []
        f_output_list = []
        rnn_tm1_list = []

        notdone = (~inputs["done"]).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            # Reset core state to zero whenever an episode ended.
            # Make `done` broadcastable with (num_layers, B, hidden_size)
            # learner's `core_state` only contains the cell state
            nd = nd.view(1, -1, 1)
            core_state = nest.map(nd.mul, core_state)
            # core_state = core_state[0]  # untuple
            rnn_tm1_list.append(core_state)  # store c(t-1)

            output, z_out, f_out, core_state = self.core(
                input.unsqueeze(0), core_state, is_actor=False)  # DONE make this in the layer definition
            # core_state = (core_state,)

            core_output_list.append(output)
            z_output_list.append(z_out)
            f_output_list.append(f_out)
        # new_cells = torch.cat(core_cell_list)  # output already has a seq dim at 0.

        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        core_output.retain_grad()

        # apply output gate
        gate_out = self.output_gate(torch.cat([core_input.view(T * B, -1), core_output], dim=-1))
        gate_out = torch.sigmoid(gate_out)
        gate_out = core_output * gate_out

        policy_logits = self.policy(gate_out)
        baseline = self.baseline(gate_out)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        # prepare "states" needed for RTRL
        rnn_tm1 = torch.cat(rnn_tm1_list)
        z_outs = torch.cat(z_output_list)  # z already has a seq dim at 0
        f_outs = torch.cat(f_output_list)  # idem

        # core_state = (new_cells, rnn_tm1, z_outs, f_outs)
        core_state = (core_input, notdone, core_output, rnn_tm1, z_outs, f_outs)

        return (action, policy_logits, baseline), core_state

    # this is the generic forward, input/output state is consistently the last (rnn_state, rtrl_state)
    def forward(self, inputs, core_state):
        x = inputs["frame"]

        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)

        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs["done"]).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            # Reset core state to zero whenever an episode ended.
            # Make `done` broadcastable with (num_layers, B, hidden_size)
            # states:
            # core_state contains tensors of shape either (num_layers, B, hidden_size) or (num_layers, B, hidden_size, hidden_size)
            hidden_prev, rtrl_state = core_state
            Z_state, F_state, wz_state, wf_state, bz_state, bf_state = rtrl_state
            nd = nd.view(1, -1, 1)
            hidden_prev = nest.map(nd.mul, hidden_prev)
            wz_state = nest.map(nd.mul, wz_state)
            wf_state = nest.map(nd.mul, wf_state)
            bz_state = nest.map(nd.mul, bz_state)
            bf_state = nest.map(nd.mul, bf_state)

            nd = nd.view(1, -1, 1, 1)
            Z_state = nest.map(nd.mul, Z_state)
            F_state = nest.map(nd.mul, F_state)

            rtrl_state = (Z_state, F_state, wz_state, wf_state, bz_state, bf_state)
            core_state = (hidden_prev, rtrl_state)

            # core_state = nest.map(nd.mul, core_state)
            # core_state = core_state[0]  # untuple
            output, core_state = self.core(input.unsqueeze(0), core_state)
            # core_state = (core_state,)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)

        # apply output gate
        gate_out = self.output_gate(torch.cat([core_input.view(T * B, -1), core_output], dim=-1))
        gate_out = torch.sigmoid(gate_out)
        gate_out = core_output * gate_out

        policy_logits = self.policy(gate_out)
        baseline = self.baseline(gate_out)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (action, policy_logits, baseline), core_state
