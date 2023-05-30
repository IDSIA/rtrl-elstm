# Reinforcement Learning Experiments using Real-Time Recurrent Actor-Critic Method (R2AC)

This repository is originally forked from [IDSIA/modern-srwm/reinforcement_learning](https://github.com/IDSIA/modern-srwm/tree/main/reinforcement_learning) which itself is a fork of the public PyTorch implementation of IMPALA, [Torchbeast](https://github.com/facebookresearch/torchbeast).

## Requirements
* We use the `Polybeast` version of [Torchbeast](https://github.com/facebookresearch/torchbeast).
* [DMLab](https://github.com/deepmind/lab) and [ProcGen](https://github.com/openai/procgen).

We refer to instructions in the original repositories to install these packages. Please check the corresponding requirements.
Note that intalling Polybeast or DMLab might not be straightforward depending on your system.

* Optionally: install `wandb` for monitoring jobs (by using the `--use_wandb` flag)

* We used PyTorch version `>= 1.4.0` for our experiments

## Training

**NB: in the code, "eLSTM" (the name of our RNN architecture in the paper) is called "QuasiLSTM".**

We have a separate main training code file for each environment: DMLab, ProcGen, and Atari.
See example scripts below.
`--use_rtrl` flag should be removed to train a feedforward agent,
or it should be replaced by `--use_quasi_lstm` for the TBPTT-trained eLSTM.

Logs of our experiments/figures (~3 GB uncompressed) can be downloaded from [here/google-drive](https://drive.google.com/file/d/1d4EhyGzVMEILZdeIMXnE_7-OfeW8yrrR/view?usp=sharing).

### DMLab

* Training from scratch:
```
SAVE_DIR=saved_models_dmlab

GAME=rooms_keys_doors_puzzle
MODEL=rtrl_elstm
SEED=
LEN=100
SIZE=512

python -m torchbeast_dmlab.polybeast \
     --single_gpu \
     --use_wandb \
     --use_rtrl \
     --seed ${SEED} \
     --env ${GAME} \
     --pipes_basename "unix:/tmp/pb_rgbdmlab_${MODEL}_${GAME}_${SIZE}_len${LEN}_seed${SEED}" \
     --validate_every 240000 \
     --disable_validation \
     --validate_step_every 1_000_000 \
     --num_actors 48 \
     --num_servers 48 \
     --total_steps 100_000_000 \
     --learning_rate 0.0006 \
     --grad_norm_clipping 40 \
     --epsilon 0.01 \
     --entropy_cost 0.01 \
     --batch_size 32 \
     --unroll_length ${LEN} \
     --hidden_size ${SIZE} \
     --num_learner_threads 1 \
     --num_inference_threads 1 \
     --project_name "2023_${GAME}" \
     --xpid "rgb_${MODEL}_${GAME}_${MODE}_${SIZE}_len${LEN}_seed${SEED}" \
     --savedir ${SAVE_DIR}
```

* Training using a pre-trained vision stem:
```
SAVE_DIR=saved_models_dmlab

GAME=rooms_watermaze
MODEL=frozen_rtrl_elstm
SEED=
LEN=100
SIZE=512

PRETRAIN='pretrained_models/me/model.tar'

python -m torchbeast_dmlab.polybeast \
     --single_gpu \
     --use_rtrl \
     --use_wandb \
     --load_conv_net_from ${PRETRAIN} \
     --freeze_conv \
     --freeze_fc \
     --env ${GAME} \
     --pipes_basename "unix:/tmp/pb_rgbdmlab_${MODEL}_${GAME}_${SIZE}_len${LEN}_seed${SEED}" \
     --validate_every 240000 \
     --disable_validation \
     --validate_step_every 1_000_000 \
     --num_actors 48 \
     --num_servers 48 \
     --total_steps 100_000_000 \
     --learning_rate 0.0006 \
     --grad_norm_clipping 40 \
     --epsilon 0.01 \
     --entropy_cost 0.01 \
     --batch_size 32 \
     --unroll_length ${LEN} \
     --hidden_size ${SIZE} \
     --num_learner_threads 1 \
     --num_inference_threads 1 \
     --project_name "2023_${GAME}_frozen" \
     --xpid "rgb_${MODEL}_${GAME}_${MODE}_${SIZE}_len${LEN}_seed${SEED}" \
     --savedir ${SAVE_DIR}
```

### ProcGen
```
SAVE_DIR=saved_models

GAME=chaser
MODE=hard
MODEL=rtrl_quasi_lstm
SEED=
LEN=50
SIZE=256

python -m torchbeast_procgen.polybeast \
     --single_gpu \
     --use_wandb \
     --use_rtrl \
     --env procgen:procgen-${GAME}-v0 \
     --pipes_basename "unix:/tmp/pb_${MODEL}_${GAME}_${SIZE}_len${LEN}_seed${SEED}" \
     --validate_every 60 \
     --num_actors 48 \
     --num_servers 48 \
     --total_steps 200_000_000 \
     --save_extra_checkpoint 50_000_000 \
     --learning_rate 0.0006 \
     --grad_norm_clipping 40 \
     --epsilon 0.01 \
     --entropy_cost 0.01 \
     --batch_size 32 \
     --unroll_length ${LEN} \
     --num_actions 15 \
     --hidden_size ${SIZE} \
     --num_learner_threads 1 \
     --num_inference_threads 1 \
     --project_name "2023_${GAME}" \
     --xpid "${MODEL}_${GAME}_${MODE}_${SIZE}_len${LEN}_seed${SEED}" \
     --num_levels 500 \
     --start_level 0 \
     --distribution_mode ${MODE} \
     --valid_distribution_mode ${MODE} \
     --valid_num_levels 500 \
     --valid_start_level 500 \
     --valid_num_episodes 10 \
     --savedir ${SAVE_DIR}
```

### Atari
For Atari, the number of actions should be changed for each game; see `list_atari_games.txt`
```
SAVE_DIR=saved_models_atari

GAME=QbertNoFrameskip-v4
MODEL=rtrl_quasi_lstm
SEED=
LEN=50
SIZE=256

python -m torchbeast_atari.polybeast \
     --single_gpu \
     --env ${GAME} \
     --use_rtrl \
     --use_wandb \
     --pipes_basename "unix:/tmp/pb_${MODEL}_${GAME}_${SIZE}_len${LEN}_seed${SEED}" \
     --disable_validation \
     --validate_every 60 \
     --num_actors 48 \
     --num_servers 48 \
     --disable_validation \
     --validate_every 6000 \
     --validate_step_every 10_000_000 \
     --total_steps 200_000_000 \
     --save_extra_checkpoint 50_000_000 \
     --learning_rate 0.0006 \
     --grad_norm_clipping 40 \
     --epsilon 0.01 \
     --entropy_cost 0.01 \
     --batch_size 32 \
     --unroll_length ${LEN} \
     --num_actions 6 \
     --hidden_size ${SIZE} \
     --num_learner_threads 1 \
     --num_inference_threads 1 \
     --project_name "2023_${GAME}" \
     --xpid "${MODEL}_${GAME}_${MODE}_${SIZE}_len${LEN}_seed${SEED}" \
     --savedir ${SAVE_DIR}
```

## Evaluation
**The final evalution is automatically carried out at the end of training.**
NB: `polybeast_learner.py` files typically also contain code for the eval-only mode; please ignore them; they are just copy-pasted from some random environments and not adapted to each environment.
