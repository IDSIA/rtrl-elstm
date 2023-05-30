# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import collections
import logging
import os
import threading
import time
import timeit
import traceback

import nest
import torch
import libtorchbeast
from torch import nn
from torch.nn import functional as F
from torchbeast.core import file_writer
from torchbeast.core import vtrace

from torchbeast_atari.model import Net
from torchbeast_atari.model import QuasiLSTMNet, RTRLQuasiLSTMNet

# Necessary for multithreading.
os.environ["OMP_NUM_THREADS"] = "1"


# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

parser.add_argument("--pipes_basename", default="unix:/tmp/polybeast",
                    help="Basename for the pipes for inter-process communication. "
                    "Has to be of the type unix:/some/path.")
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")

# Training settings.
parser.add_argument("--single_gpu", action="store_true",
                    help="use single gpu.")
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--disable_validation", action="store_true",
                    help="Disable validation.")
parser.add_argument("--validate_every", default=10, type=int,
                    help="run validation every this *minutes*.")
parser.add_argument("--validate_step_every", default=-1, type=int,
                    help="run validation every this *steps*.")
parser.add_argument("--save_extra_checkpoint", default=50000000, type=int,
                    help="Save an extra checkpoint at .")
parser.add_argument("--eval_extra", action="store_true",
                    help="Eval extra checkpoint.")
parser.add_argument("--savedir", default="~/palaas/torchbeast",
                    help="Root dir where experiment data will be saved.")

parser.add_argument("--load_conv_net_from", help="Path to the checkpoint to load.")
parser.add_argument("--not_load_fc", action="store_true",
                    help="Load conv net but not FC layer.")
parser.add_argument("--freeze_conv", action="store_true",
                    help="do not update convolutional vision stem.")
parser.add_argument("--freeze_fc", action="store_true",
                    help="do not update fc after convolutional vision stem.")
parser.add_argument("--use_rtrl", action="store_true",
                    help="train with RTRL.")
parser.add_argument("--num_actors", default=4, type=int, metavar="N",
                    help="Number of actors.")
parser.add_argument("--total_steps", default=100000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=8, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=80, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--num_learner_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--num_inference_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--num_actions", default=6, type=int, metavar="A",
                    help="Number of actions.")
parser.add_argument("--use_lstm", action="store_true",
                    help="Use LSTM in agent model.")
parser.add_argument("--use_quasi_lstm", action="store_true",
                    help="Use Quasi LSTM in agent model.")
parser.add_argument("--use_delta_rnn", action="store_true",
                    help="Use Delta RNN in agent model.")
parser.add_argument("--use_delta", action="store_true",
                    help="Use Delta Net in agent model.")
parser.add_argument("--use_lt", action="store_true",
                    help="Use Linear Trafo in agent model.")
parser.add_argument("--use_rec_delta", action="store_true",
                    help="Use Recurrent Delta Net in agent model.")
parser.add_argument("--use_dd", action="store_true",
                    help="Use Delta Delta in agent model.")
parser.add_argument("--use_xem", action="store_true",
                    help="Use cross episode memory when DD is the agent.")
parser.add_argument("--keep_dd", action="store_true",
                    help="Keep delta delta XEM as part of the model.")
parser.add_argument("--max_learner_queue_size", default=None, type=int, metavar="N",
                    help="Optional maximum learner queue size. Defaults to batch_size.")

# Model settings.
parser.add_argument("--hidden_size", default=256, type=int,
                    help="transformer or LSTM hidden size.")  # WARNING, before it was 128. FINDME
parser.add_argument("--dim_ff", default=512, type=int,
                    help="transformer hidden size.")
parser.add_argument("--dim_head", default=32, type=int,
                    help="transformer head size.")
parser.add_argument("--num_layers", default=2, type=int,
                    help="tranformer num layers.")
parser.add_argument("--num_head", default=4, type=int,
                    help="tranformer num heads.")
parser.add_argument("--dropout", default=0.0, type=float,
                    help="tranformer dropout.")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.0006, type=float,
                    help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5, type=float,
                    help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99, type=float,
                    help="Discounting factor.")
parser.add_argument("--reward_clipping", default="abs_one",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--learning_rate", default=0.00048, type=float,
                    metavar="LR", help="Learning rate.")
parser.add_argument("--alpha", default=0.99, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0, type=float,
                    help="RMSProp momentum.")
parser.add_argument("--epsilon", default=0.01, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")

# Misc settings.
parser.add_argument("--write_profiler_trace", action="store_true",
                    help="Collect and write a profiler trace "
                    "for chrome://tracing/.")
# yapf: enable

parser.add_argument('--num_servers', default=4, type=int, metavar='N',
                    help='Number of environment servers.')
parser.add_argument('--env', type=str, default='PongNoFrameskip-v4',
                    help='Gym environment.')
parser.add_argument('--allow_oov', action="store_true",
                    help='Allow action space larger than the env specific one.'
                    ' All out-of-vocab action will be mapped to NoOp.')
parser.add_argument('--valid_num_episodes', default=5, type=int, metavar='N',
                    help='number of validation episodes.')
parser.add_argument('--valid_num_runs', default=1, type=int, metavar='N',
                    help='number of validation runs '
                         '(each run is valid_num_episodes episode long).')

# Wandb settings
parser.add_argument('--project_name', type=str, default=None,
                    help='project name for wandb.')
parser.add_argument('--job_name', type=str, default=None,
                    help='job name for wandb.')
parser.add_argument('--use_wandb', action='store_true',
                    help='use wandb.')

args = parser.parse_args()

if args.use_wandb:  # configure wandb.
    import wandb
    use_wandb = True

    if args.project_name is None:
        project_name = (os.uname()[1]
                        + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        project_name = args.project_name

    wandb.init(project=project_name)

    if args.job_name is None:
        # wandb.run.name = (os.uname()[1]
        #                   + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        #                   + args.work_dir)
        wandb.run.name = f"{os.uname()[1]}" \
                         f"-{args.mode}" \
                         f"-{args.xpid}" \
                         f"-{args.disable_checkpoint}" \
                         f"-{args.savedir}" \
                         f"-{args.num_actors}" \
                         f"-{args.total_steps}" \
                         f"-{args.batch_size}" \
                         f"-{args.unroll_length}" \
                         f"-{args.hidden_size}" \
                         f"-{args.entropy_cost}" \
                         f"-{args.baseline_cost}" \
                         f"-{args.discounting}" \
                         f"-{args.reward_clipping}" \
                         f"-{args.learning_rate}" \
                         f"-{args.alpha}" \
                         f"-{args.momentum}" \
                         f"-{args.epsilon}" \
                         f"-{args.grad_norm_clipping}"
    else:
        wandb.run.name = f"{os.uname()[1]}//{args.job_name}"

    config = wandb.config
    config.host = os.uname()[1]  # host node name
    config.mode=args.mode
    config.xpid=args.xpid
    config.disable_checkpoint=args.disable_checkpoint
    config.savedir=args.savedir
    config.num_actors=args.num_actors
    config.total_steps=args.total_steps
    config.batch_size=args.batch_size
    config.unroll_length=args.unroll_length
    config.hidden_size=args.hidden_size
    config.disable_cuda=args.disable_cuda
    config.use_lstm=args.use_lstm
    config.use_quasi_lstm=args.use_quasi_lstm
    config.entropy_cost=args.entropy_cost
    config.baseline_cost=args.baseline_cost
    config.discounting=args.discounting
    config.reward_clipping=args.reward_clipping
    config.learning_rate=args.learning_rate
    config.alpha=args.alpha
    config.momentum=args.momentum
    config.epsilon=args.epsilon
    config.grad_norm_clipping=args.grad_norm_clipping
    config.freeze_conv=args.freeze_conv
    config.freeze_fc=args.freeze_fc
    config.load_conv_net_from=args.load_conv_net_from
    config.not_load_fc=args.not_load_fc
    config.use_rtrl=args.use_rtrl
else:
    use_wandb = False


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def inference(flags, inference_batcher, model, lock=threading.Lock()):  # noqa: B008
    with torch.no_grad():
        for batch in inference_batcher:
            batched_env_outputs, agent_state = batch.get_inputs()
            frame, reward, done, *_ = batched_env_outputs
            frame = frame.to(flags.actor_device, non_blocking=True)
            reward = reward.to(flags.actor_device, non_blocking=True)
            done = done.to(flags.actor_device, non_blocking=True)
            agent_state = nest.map(
                lambda t: t.to(flags.actor_device, non_blocking=True), agent_state
            )
            with lock:
                outputs = model(
                    dict(frame=frame, reward=reward, done=done), agent_state
                )
            outputs = nest.map(lambda t: t.cpu(), outputs)
            batch.set_outputs(outputs)


EnvOutput = collections.namedtuple(
    "EnvOutput", "frame rewards done episode_step episode_return"
)
AgentOutput = collections.namedtuple("AgentOutput", "action policy_logits baseline")
Batch = collections.namedtuple("Batch", "env agent")


def learn(
    flags,
    learner_queue,
    model,
    actor_model,
    optimizer,
    scheduler,
    stats,
    plogger,
    lock=threading.Lock(),
):
    for tensors in learner_queue:
        tensors = nest.map(lambda t: t.to(flags.learner_device), tensors)

        batch, initial_agent_state = tensors
        env_outputs, actor_outputs = batch
        frame, reward, done, *_ = env_outputs

        lock.acquire()  # Only one thread learning at a time.
        learner_outputs, unused_state = model(
            dict(frame=frame, reward=reward, done=done), initial_agent_state
        )

        # Take final value function slice for bootstrapping.
        learner_outputs = AgentOutput._make(learner_outputs)
        bootstrap_value = learner_outputs.baseline[-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = nest.map(lambda t: t[1:], batch)
        learner_outputs = nest.map(lambda t: t[:-1], learner_outputs)

        # Turn into namedtuples again.
        env_outputs, actor_outputs = batch
        env_outputs = EnvOutput._make(env_outputs)
        actor_outputs = AgentOutput._make(actor_outputs)
        learner_outputs = AgentOutput._make(learner_outputs)

        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(env_outputs.rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = env_outputs.rewards

        discounts = (~env_outputs.done).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=actor_outputs.policy_logits,
            target_policy_logits=learner_outputs.policy_logits,
            actions=actor_outputs.action,
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs.baseline,
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs.policy_logits,
            actor_outputs.action,
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs.baseline
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs.policy_logits
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())

        episode_returns = env_outputs.episode_return[env_outputs.done]
        stats["step"] = stats.get("step", 0) + flags.unroll_length * flags.batch_size
        stats["episode_returns"] = tuple(episode_returns.cpu().numpy())
        stats["mean_episode_return"] = torch.mean(episode_returns).item()
        stats["mean_episode_step"] = torch.mean(env_outputs.episode_step.float()).item()
        stats["total_loss"] = total_loss.item()
        stats["pg_loss"] = pg_loss.item()
        stats["baseline_loss"] = baseline_loss.item()
        stats["entropy_loss"] = entropy_loss.item()

        stats["learner_queue_size"] = learner_queue.size()

        if use_wandb:
            wandb.log({"episode_returns": stats["episode_returns"]})
            wandb.log({"mean_episode_step": stats["mean_episode_step"]})
            wandb.log({"mean_episode_return": stats["mean_episode_return"]})
            wandb.log({"total_loss": stats["total_loss"]})
            wandb.log({"pg_loss": stats["pg_loss"]})
            wandb.log({"baseline_loss": stats["baseline_loss"]})
            wandb.log({"entropy_loss": stats["entropy_loss"]})

        plogger.log(stats)

        if not len(episode_returns):
            # Hide the mean-of-empty-tuple NaN as it scares people.
            stats["mean_episode_return"] = None

        lock.release()


def learn_rtrl(
    flags,
    learner_queue,
    model,
    actor_model,
    optimizer,
    scheduler,
    stats,
    plogger,
    lock=threading.Lock(),
):
    for tensors in learner_queue:
        tensors = nest.map(lambda t: t.to(flags.learner_device), tensors)

        batch, initial_agent_state = tensors
        env_outputs, actor_outputs = batch
        frame, reward, done, *_ = env_outputs

        lock.acquire()  # Only one thread learning at a time.
        # forward in learner mode, get various needed states
        learner_state, _ = initial_agent_state
        learner_outputs, learner_state = model.forward_learner(
            dict(frame=frame, reward=reward, done=done), learner_state
        )

        # Take final value function slice for bootstrapping.
        learner_outputs = AgentOutput._make(learner_outputs)
        bootstrap_value = learner_outputs.baseline[-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = nest.map(lambda t: t[1:], batch)
        learner_outputs = nest.map(lambda t: t[:-1], learner_outputs)

        # Turn into namedtuples again.
        env_outputs, actor_outputs = batch
        env_outputs = EnvOutput._make(env_outputs)
        actor_outputs = AgentOutput._make(actor_outputs)
        learner_outputs = AgentOutput._make(learner_outputs)

        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(env_outputs.rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = env_outputs.rewards

        discounts = (~env_outputs.done).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=actor_outputs.policy_logits,
            target_policy_logits=learner_outputs.policy_logits,
            actions=actor_outputs.action,
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs.baseline,
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs.policy_logits,
            actor_outputs.action,
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs.baseline
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs.policy_logits
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        optimizer.zero_grad()
        model.core.rtrl_zero_grad()
        total_loss.backward()  # this also computes the top_gradients dL / dc_t

        # compute gradients of params trained by RTRL
        cached_input, notdone, new_cells, rnn_outs_tm1, z_outs, f_outs = learner_state  # DONE in model, DONE in layer, change ouput states for forward_learner

        _, rtrl_states = initial_agent_state  # DONE, all states should be augmented by RTRL states
        # all_states_for_rtrl = (rnn_outs_tm1, z_outs, f_outs, rtrl_states)
        Z_state, F_state, wz_state, wf_state, bz_state, bf_state = rtrl_states

        # removing layer dim, non priority but better to remove this
        Z_state = Z_state.squeeze(0)
        F_state = F_state.squeeze(0)
        wz_state = wz_state.squeeze(0)
        wf_state = wf_state.squeeze(0)
        bz_state = bz_state.squeeze(0)
        bf_state = bf_state.squeeze(0)

        rtrl_states = (Z_state, F_state, wz_state, wf_state, bz_state, bf_state)

        slen, bsz, _ = rnn_outs_tm1.shape

        model.compute_grad_rtrl(cached_input, notdone, rnn_outs_tm1, z_outs, f_outs,
                                rtrl_states, new_cells.grad.view(slen, bsz, -1))
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())

        episode_returns = env_outputs.episode_return[env_outputs.done]
        stats["step"] = stats.get("step", 0) + flags.unroll_length * flags.batch_size
        stats["episode_returns"] = tuple(episode_returns.cpu().numpy())
        stats["mean_episode_return"] = torch.mean(episode_returns).item()
        stats["mean_episode_step"] = torch.mean(env_outputs.episode_step.float()).item()
        stats["total_loss"] = total_loss.item()
        stats["pg_loss"] = pg_loss.item()
        stats["baseline_loss"] = baseline_loss.item()
        stats["entropy_loss"] = entropy_loss.item()

        stats["learner_queue_size"] = learner_queue.size()

        if use_wandb:
            wandb.log({"episode_returns": stats["episode_returns"]})
            wandb.log({"mean_episode_step": stats["mean_episode_step"]})
            wandb.log({"mean_episode_return": stats["mean_episode_return"]})
            wandb.log({"total_loss": stats["total_loss"]})
            wandb.log({"pg_loss": stats["pg_loss"]})
            wandb.log({"baseline_loss": stats["baseline_loss"]})
            wandb.log({"entropy_loss": stats["entropy_loss"]})

        plogger.log(stats)

        if not len(episode_returns):
            # Hide the mean-of-empty-tuple NaN as it scares people.
            stats["mean_episode_return"] = None

        lock.release()


def train(flags):
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser(
            "%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )
    best_valid_checkpointpath = os.path.expandvars(
        os.path.expanduser(
            "%s/%s/%s" % (flags.savedir, flags.xpid, "model_best_val.tar"))
    )

    if flags.save_extra_checkpoint > 0:
        extra_checkpointpath = os.path.expandvars(
            os.path.expanduser(
                "%s/%s/%s" % (flags.savedir, flags.xpid, "model_extra.tar")))

    if flags.single_gpu:
        logging.info("Using single GPU.")
        flags.learner_device = torch.device("cuda:0")
        flags.actor_device = torch.device("cuda:0")
    elif not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.learner_device = torch.device("cuda:0")
        flags.actor_device = torch.device("cuda:1")
    else:
        logging.info("Not using CUDA.")
        flags.learner_device = torch.device("cpu")
        flags.actor_device = torch.device("cpu")

    if flags.max_learner_queue_size is None:
        flags.max_learner_queue_size = flags.batch_size

    # The queue the learner threads will get their data from.
    # Setting `minimum_batch_size == maximum_batch_size`
    # makes the batch size static.
    learner_queue = libtorchbeast.BatchingQueue(
        batch_dim=1,
        minimum_batch_size=flags.batch_size,
        maximum_batch_size=flags.batch_size,
        check_inputs=True,
        maximum_queue_size=flags.max_learner_queue_size,
    )

    # The "batcher", a queue for the inference call. Will yield
    # "batch" objects with `get_inputs` and `set_outputs` methods.
    # The batch size of the tensors will be dynamic.
    inference_batcher = libtorchbeast.DynamicBatcher(
        batch_dim=1,
        minimum_batch_size=1,
        maximum_batch_size=512,
        timeout_ms=100,
        check_outputs=True,
    )

    addresses = []
    connections_per_server = 1
    pipe_id = 0
    while len(addresses) < flags.num_actors:
        for _ in range(connections_per_server):
            addresses.append(f"{flags.pipes_basename}.{pipe_id}")
            if len(addresses) == flags.num_actors:
                break
        pipe_id += 1

    if flags.use_quasi_lstm:
        model = QuasiLSTMNet(
            num_actions=flags.num_actions,
            hidden_size=flags.hidden_size, freeze_conv=flags.freeze_conv,
            freeze_fc=flags.freeze_fc)
        logging.info(model)
        model = model.to(device=flags.learner_device)

        actor_model = QuasiLSTMNet(
            num_actions=flags.num_actions,
            hidden_size=flags.hidden_size, freeze_conv=flags.freeze_conv,
            freeze_fc=flags.freeze_fc)
        actor_model.to(device=flags.actor_device)

    elif flags.use_rtrl:
        model = RTRLQuasiLSTMNet(
            num_actions=flags.num_actions,
            hidden_size=flags.hidden_size, freeze_conv=flags.freeze_conv,
            freeze_fc=flags.freeze_fc)
        logging.info(model)
        model = model.to(device=flags.learner_device)

        actor_model = RTRLQuasiLSTMNet(
            num_actions=flags.num_actions,
            hidden_size=flags.hidden_size, freeze_conv=flags.freeze_conv,
            freeze_fc=flags.freeze_fc)
        actor_model.to(device=flags.actor_device)
    else:
        model = Net(num_actions=flags.num_actions,
                    use_lstm=flags.use_lstm, hidden_size=flags.hidden_size,
                    freeze_conv=flags.freeze_conv, freeze_fc=flags.freeze_fc)
        logging.info(model)
        model = model.to(device=flags.learner_device)

        actor_model = Net(
            num_actions=flags.num_actions,
            use_lstm=flags.use_lstm, hidden_size=flags.hidden_size,
            freeze_conv=flags.freeze_conv, freeze_fc=flags.freeze_fc)
        actor_model.to(device=flags.actor_device)

    # The ActorPool that will run `flags.num_actors` many loops.
    actors = libtorchbeast.ActorPool(
        unroll_length=flags.unroll_length,
        learner_queue=learner_queue,
        inference_batcher=inference_batcher,
        env_server_addresses=addresses,
        initial_agent_state=actor_model.initial_state(),
    )

    def run():
        try:
            actors.run()
        except Exception as e:
            logging.error("Exception in actorpool thread!")
            traceback.print_exc()
            print()
            raise e

    actorpool_thread = threading.Thread(target=run, name="actorpool-thread")

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return (
            1
            - min(epoch * flags.unroll_length * flags.batch_size, flags.total_steps)
            / flags.total_steps
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    stats = {}

    best_valid_return = 0

    # Load state from a checkpoint, if possible.
    if os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath, map_location=flags.learner_device
        )
        model.load_state_dict(checkpoint_states["model_state_dict"])
        optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_states["scheduler_state_dict"])
        stats = checkpoint_states["stats"]
        best_valid_return = checkpoint_states["best_valid_return"]
        logging.info(f"Resuming preempted job, current stats:\n{stats}")
    # load pre-trained conv net from a pre-trained checkpoint
    elif args.load_conv_net_from is not None:
        checkpoint_states = torch.load(
            args.load_conv_net_from, map_location=flags.learner_device
        )
        # Load only vision stem params, i.e.
        # self.feat_convs
        # self.resnet1
        # self.resnet2
        # self.fc
        logging.info(f"Loading ConvNet from: {args.load_conv_net_from}")
        model_dict = model.state_dict()
        pre_trained_dict = checkpoint_states["model_state_dict"]
        vision_only_dict = {}
        if args.not_load_fc:  # only load conv
            for key, value in pre_trained_dict.items():
                if 'feat_convs' in key or 'resnet' in key:
                    vision_only_dict[key] = value
        else:
            for key, value in pre_trained_dict.items():
                if 'feat_convs' in key or 'resnet' in key or 'fc' in key:
                    vision_only_dict[key] = value
        model_dict.update(vision_only_dict)
        model.load_state_dict(model_dict)

    # Initialize actor model like learner model.
    actor_model.load_state_dict(model.state_dict())

    if args.use_rtrl:
        learn_fn = learn_rtrl
    else:
        learn_fn = learn

    learner_threads = [
        threading.Thread(
            target=learn_fn,
            name="learner-thread-%i" % i,
            args=(
                flags,
                learner_queue,
                model,
                actor_model,
                optimizer,
                scheduler,
                stats,
                plogger,
            ),
        )
        for i in range(flags.num_learner_threads)
    ]
    inference_threads = [
        threading.Thread(
            target=inference,
            name="inference-thread-%i" % i,
            args=(flags, inference_batcher, actor_model),
        )
        for i in range(flags.num_inference_threads)
    ]

    actorpool_thread.start()
    for t in learner_threads + inference_threads:
        t.start()

    def checkpoint(path):
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", path)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "stats": stats,
                "flags": vars(flags),
                "best_valid_return": best_valid_return,
            },
            path,
        )

    # validate in all env
    def validate(num_episodes=5, num_runs=1, plogger=None, step=0):
        from torchbeast.core import environment
        from torchbeast_atari_noframestack import atari_wrappers

        def create_test_env(flags):
            return atari_wrappers.wrap_pytorch(
                atari_wrappers.wrap_deepmind(
                    atari_wrappers.make_atari(flags.env),
                    clip_rewards=False,
                    frame_stack=True,
                    scale=False,
                )
            )

        env_name = flags.env
        gym_env = create_test_env(flags)
        env = environment.Environment(gym_env)

        # model.eval()
        device = flags.learner_device  # move to actor device?? which should be less busy? and copy back

        observation = env.initial()
        all_returns = []

        core_state = model.initial_state()
        core_state = nest.map(lambda t: t.to(device), core_state)

        logging.info(f"Start validation on {env_name}")
        with torch.no_grad():
            while len(all_returns) < num_runs:
                returns = []
                while len(returns) < num_episodes:
                    # (action, policy_logits, baseline), core_state
                    observation = nest.map(
                        lambda t: t.to(device), observation)
                    agent_outputs, core_state = model(
                        observation, core_state)
                    action, _, _ = agent_outputs
                    observation = env.step(action)
                    if observation["done"].item():
                        returns.append(
                            observation["episode_return"].item())
                        logging.info(
                            "Episode ended after %d steps. Return: %.1f",
                            observation["episode_step"].item(),
                            observation["episode_return"].item(),
                        )
                logging.info(
                    "Average returns over %i episodes: %.1f",
                        num_episodes, sum(returns) / len(returns)
                )
                all_returns.append(sum(returns) / len(returns))
        env.close()
        import numpy as np

        if num_runs == 1:
            mean_valid_return = np.mean(returns)
            plogger._logger.info(
                f"[Step {step} validation, {env_name}] Average returns over "
                f"{num_episodes} for {num_runs} runs: {returns}")
            plogger._logger.info(
                f"[Step {step} validation, {env_name}] Mean return: "
                f"{mean_valid_return:.1f}, std: {np.std(returns):.1f}")
        else:
            mean_valid_return = np.mean(all_returns)
            plogger._logger.info(
                f"[Step {step} validation, {env_name}] Average returns over "
                f"{num_episodes} for {num_runs} runs: {all_returns}")
            plogger._logger.info(
                f"[Step {step} validation, {env_name}] Mean return: "
                f"{mean_valid_return:.1f}, std: {np.std(all_returns):.1f}")
        if use_wandb:
            wandb.log({f"{env_name} validation return": mean_valid_return})

        model.train()

        return mean_valid_return

    def format_value(x):
        return f"{x:1.5}" if isinstance(x, float) else str(x)

    try:
        if flags.save_extra_checkpoint > 0:
            saved_extra = False
        last_checkpoint_time = timeit.default_timer()
        while True:
            start_time = timeit.default_timer()
            start_step = stats.get("step", 0)

            if start_step >= flags.total_steps:
                break
            time.sleep(5)
            end_step = stats.get("step", 0)

            if timeit.default_timer() - last_checkpoint_time > flags.validate_every * 60 or start_step % flags.validate_step_every == 0:
                # Validate every 10 min.
                if not flags.disable_validation:
                    val_start = timeit.default_timer()
                    mean_valid_return = validate(
                        num_episodes=flags.valid_num_episodes,
                        num_runs=flags.valid_num_runs, plogger=plogger, step=end_step)
                    val_dur = val_start - timeit.default_timer()
                    logging.info(f"[validation] duration: "
                                 f"{val_dur:.1f} sec.")
                    if use_wandb:
                        wandb.log({"validation time": val_dur})
                    if mean_valid_return > best_valid_return:
                        # save if best
                        best_valid_return = mean_valid_return
                        checkpoint(best_valid_checkpointpath)
                # Always save latest checkpoint
                checkpoint(checkpointpath)
                last_checkpoint_time = timeit.default_timer()

            logging.info(
                "Step %i @ %.1f SPS. Inference batcher size: %i."
                " Learner queue size: %i."
                " Other stats: (%s)",
                end_step,
                (end_step - start_step) / (timeit.default_timer() - start_time),
                inference_batcher.size(),
                learner_queue.size(),
                ", ".join(
                    f"{key} = {format_value(value)}" for key, value in stats.items()
                ),
            )
            if flags.save_extra_checkpoint > 0:
                if saved_extra is False and end_step > flags.save_extra_checkpoint:
                    logging.info(f"Step {end_step} Saving EXTRA checkpoint to {extra_checkpointpath}")
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "stats": stats,
                            "flags": vars(flags),
                            "best_valid_return": best_valid_return,
                        },
                        extra_checkpointpath,
                    )
                    saved_extra = True

    except KeyboardInterrupt:
        pass  # Close properly.
    else:
        logging.info("Learning finished after %i steps.", stats["step"])
        checkpoint(checkpointpath)

    # Done with learning. Stop all the ongoing work.
    inference_batcher.close()
    learner_queue.close()

    actorpool_thread.join()

    for t in learner_threads + inference_threads:
        t.join()

    full_test(flags, checkpointpath, plogger)
    logging.info(f'End of training after {stats["step"]} steps.')


# test on train, and three test sets
def full_test(flags, checkpointpath, plogger, num_runs=5, num_episodes=30, device='cuda'):
    plogger._logger.info("============= Final Evaluation ===============")

    import numpy as np
    from torchbeast_atari_noframestack import atari_wrappers
    from torchbeast.core import environment

    def create_test_env(flags):
        return atari_wrappers.wrap_pytorch(
            atari_wrappers.wrap_deepmind(
                atari_wrappers.make_atari(flags.env),
                clip_rewards=False,
                frame_stack=True,
                scale=False,
            )
        )

    gym_env = create_test_env(flags)
    env = environment.Environment(gym_env)

    # define model and load params from the checkpoint
    if flags.use_quasi_lstm:
        model = QuasiLSTMNet(
            num_actions=flags.num_actions, hidden_size=flags.hidden_size)
    elif flags.use_rtrl:
        model = RTRLQuasiLSTMNet(
            num_actions=flags.num_actions,
            hidden_size=flags.hidden_size, freeze_conv=flags.freeze_conv,
            freeze_fc=flags.freeze_fc)
    else:
        model = Net(num_actions=flags.num_actions,
                    use_lstm=flags.use_lstm, hidden_size=flags.hidden_size)

    plogger._logger.info(model)
    plogger._logger.info(f"# params: "
                         f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model = model.to(device)
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_results = {}
    final_results_list = []

    logging.info("Start eval")
    for start_level in range(num_runs):
        observation = env.initial()

        core_state = model.initial_state()
        core_state = nest.map(lambda t: t.to(device), core_state)

        plogger._logger.info(f"Start eval {num_episodes} episodes for run {start_level}")
        with torch.no_grad():
            returns = []
            ep_counter = 0
            while len(returns) < num_episodes:
                if flags.mode == "test_render":
                    env.gym_env.render()
                # (action, policy_logits, baseline), core_state
                observation = nest.map(lambda t: t.to(device), observation)
                agent_outputs, core_state = model(observation, core_state)
                action, _, _ = agent_outputs
                observation = env.step(action)
                if observation["done"].item():
                    ep_counter += 1
                    returns.append(observation["episode_return"].item())
                    plogger._logger.info(
                        f'[{start_level}] Episode {ep_counter} ended after {observation["episode_step"].item()} steps. Return: {observation["episode_return"].item() :.1f}',
                    )
            plogger._logger.info(
                f"Average returns over {num_episodes} episodes: {sum(returns) / len(returns):.1f}"
            )
        env.close()
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        plogger._logger.info(
            f"[From level {start_level}] Average returns over "
            f"{num_episodes} episodes: {returns}")
        plogger._logger.info(
            f"[From level {start_level}] Mean return: "
            f"{mean_return:.1f}, std: {std_return:.1f}")

        result = {'mean': mean_return, 'std': std_return}
        final_results[start_level] = result
        final_results_list.append(mean_return)

    plogger._logger.info("============= End of all evaluations ===============")
    plogger._logger.info(f"Result summary:\n {final_results}")
    plogger._logger.info(f"Train: {final_results_list[0]:.1f}, std: {final_results_list[0]:.1f} ")
    plogger._logger.info(f"Mean over 3 test sets: {np.mean(final_results_list[1:]):.1f}, std: {np.std(final_results_list[1:]):.1f} ")


# ignore; not used
def test(flags, num_episodes=30, num_runs=5, device='cuda'):
    if flags.xpid is None:
        checkpointpath = "./latest/model.tar"
    elif flags.eval_extra:
        checkpointpath = os.path.expandvars(
            os.path.expanduser(
                "%s/%s/%s" % (flags.savedir, flags.xpid, "model_extra.tar"))
        )
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser(
                "%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
        )

    from torchbeast_atari_noframestack import atari_wrappers
    from torchbeast.core import environment

    def create_test_env(flags):
        return atari_wrappers.wrap_pytorch(
            atari_wrappers.wrap_deepmind(
                atari_wrappers.make_atari(flags.env),
                clip_rewards=False,
                frame_stack=True,
                scale=False,
            )
        )

    gym_env = create_test_env(flags)
    env = environment.Environment(gym_env)

    if flags.use_quasi_lstm:
        model = QuasiLSTMNet(
            num_actions=flags.num_actions, hidden_size=flags.hidden_size)

    elif flags.use_rtrl:
        model = RTRLQuasiLSTMNet(
            num_actions=flags.num_actions,
            hidden_size=flags.hidden_size, freeze_conv=flags.freeze_conv)

    else:
        model = Net(num_actions=flags.num_actions,
                    use_lstm=flags.use_lstm, hidden_size=flags.hidden_size)

    print(model)
    print(f"# params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model = model.to(device)
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    observation = env.initial()
    # returns = []
    all_returns = []

    core_state = model.initial_state()
    core_state = nest.map(lambda t: t.to(device), core_state)

    logging.info("Start eval")
    with torch.no_grad():
        while len(all_returns) < num_runs:
            returns = []
            while len(returns) < num_episodes:
                if flags.mode == "test_render":
                    env.gym_env.render()
                # (action, policy_logits, baseline), core_state
                observation = nest.map(lambda t: t.to(device), observation)
                agent_outputs, core_state = model(observation, core_state)
                action, _, _ = agent_outputs
                # action = action.to('cpu')
                # policy_outputs, _ = agent_outputs
                observation = env.step(action)
                # observation = env.step(policy_outputs["action"])
                if observation["done"].item():
                    returns.append(observation["episode_return"].item())
                    logging.info(
                        "Episode ended after %d steps. Return: %.1f",
                        observation["episode_step"].item(),
                        observation["episode_return"].item(),
                    )
            logging.info(
                "Average returns over %i episodes: %.1f",
                    num_episodes, sum(returns) / len(returns)
            )
            all_returns.append(sum(returns) / len(returns))
    env.close()
    import numpy as np
    logging.info(f"Average returns over {num_episodes} for {num_runs} runs: {all_returns}")
    logging.info(f"{flags.env}, Mean return: {np.mean(all_returns):.1f}, std: {np.std(all_returns):.1f}")


def main(flags):
    if not flags.pipes_basename.startswith("unix:"):
        raise Exception("--pipes_basename has to be of the form unix:/some/path.")

    if flags.mode == "train":
        if flags.write_profiler_trace:
            logging.info("Running with profiler.")
            with torch.autograd.profiler.profile() as prof:
                train(flags)
            filename = "chrome-%s.trace" % time.strftime("%Y%m%d-%H%M%S")
            logging.info("Writing profiler trace to '%s.gz'", filename)
            prof.export_chrome_trace(filename)
            os.system("gzip %s" % filename)
        else:
            train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
