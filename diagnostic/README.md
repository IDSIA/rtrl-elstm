# Diagnostic Task/Copy Task

This repository was originally forked from [IDSIA/recurrent-fwp/algorithmic](https://github.com/IDSIA/recurrent-fwp/tree/master/algorithmic).

## Requirements
* PyTorch (`>= 1.6.0` recommended)

## Data Generation

```
cd utils
mkdir data_copy_task
# `code_length` is the number bits in the pattern to be memorised
# set it to 50 or 500 to get the dataset used in the paper
python copy_task_generator.py  --dump_dir data_copy_task --code_length 500
```

## Training
The training script is as follows:

`model_type` specifies the learning algo/model type:
* `0`: BPTT LSTM
* `10`: BPTT eLSTM
* `11`: RTRL eLSTM; use this with `rtrl_copy_task_main.py`

**NB: in the code, "eLSTM" (the name of our RNN architecture in the paper) is called "QuasiLSTM".**

`--level` specifies the max sequence length of the patterns to be memorised (either 50 or 500 in the paper)

```
DATA_DIR='utils/data_copy_task'

python rtrl_copy_task_main.py \
  --data_dir ${DATA_DIR} \
  --level 500 \
  --model_type 11 \
  --no_embedding \
  --num_layer 1 \
  --hidden_size 2048 \
  --dropout 0.0 \
  --batch_size 128 \
  --learning_rate 3e-5 \
  --clip 1.0 \
  --grad_cummulate 1 \
  --num_epoch 500 \
  --seed 1 \
```

`rtrl_copy_task_main.py` should be replaced by `copy_task_main.py` for all non-RTRL settings (there are many code duplications; we leave them as is).

Note that unlike prior work, we do not use any curriculum learning.

## Evaluation
Evalution is automatically run at the end of training using the best performing checkpoint based on the validation accuracy (which should be 100% for this task).

## Gradient Test
Basic implementation of RTRL forward recursion equations and the corresponding gradient test can be found in `rtrl_layers.py`
