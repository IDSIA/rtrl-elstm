# Code to generate the data set for copy task.
#
# Sketch:
# For a given max code length T,
# sample length t between 1 and T,
# randomly sample 0 or 1 for the given sequence length t to get `seq`
# create a sequence of the same length filled with `2` (i.e, special token) `2*`
# input = `seq` + `2*`
# output = `2` + `seq`

import sys
from random import randrange as drw
import random
import numpy as np

# Task hyper-parameters
rnd_seed = 42
random.seed(rnd_seed)
np.random.seed(rnd_seed)


# Get number of characters in the string w/o spaces
# NB: line break '\n' counts as one character.
def num_token(string):
    return len(string.split())


# max_seq_length is the max seq length of the pattern/code to be memorized
# for length-padding, use the same token as memory token and skip from the loss
def get_data_pair(max_seq_length, pad_id=2):
    '''Get one example of input/output pair.'''
    slen = drw(1, max_seq_length + 1)
    pattern = np.random.randint(2, size=slen)
    spaces = np.ones_like(pattern) * pad_id
    padding = np.ones([max_seq_length - slen]).astype(int) * pad_id
    input_str = np.concatenate((pattern, spaces, padding))
    tgt_str = np.concatenate((spaces, pattern, padding))

    input_str = ' '.join(map(str, input_str))
    tgt_str = ' '.join(map(str, tgt_str))

    return input_str, tgt_str

# Visualize alignment
def visualize(code_str, tgt_str):

    print("=== Code string ============ ")
    print(code_str)

    print("\n=== Target string ========== ")
    print(tgt_str)

    print("=== END ")


if __name__ == '__main__':

    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Generate data.')
    parser.add_argument('--dump_dir',
        required=True, help='where to store the data')
    parser.add_argument('--train_size', required=False, default=10000,
        type=int, help='Number of examples in the train set.')
    parser.add_argument('--valid_size', required=False, default=1000,
        type=int, help='Number of examples in the valid set.')
    parser.add_argument('--test_size', required=False, default=1000,
        type=int, help='Number of examples in the test set.')

    parser.add_argument('--code_length', required=False, default=50,
        type=int, help='Number of statements in each example.')
    parser.add_argument('--show_example', required=False, action='store_true',
        help='Only show one example.')

    args = parser.parse_args()

    in_sfx = ".src"
    out_sfx = ".tgt"

    train_file_name = f"train_{args.code_length}"
    valid_file_name = f"valid_{args.code_length}"
    test_file_name = f"test_{args.code_length}"

    tr_src = f"{args.dump_dir}/{train_file_name}{in_sfx}"
    tr_tgt = f"{args.dump_dir}/{train_file_name}{out_sfx}"

    valid_src = f"{args.dump_dir}/{valid_file_name}{in_sfx}"
    valid_tgt = f"{args.dump_dir}/{valid_file_name}{out_sfx}"

    test_src = f"{args.dump_dir}/{test_file_name}{in_sfx}"
    test_tgt = f"{args.dump_dir}/{test_file_name}{out_sfx}"

    if args.show_example:
        code_str, tgt_str = get_data_pair(args.code_length)
        visualize(code_str, tgt_str)
        sys.exit(0)

    # train
    print("Generating train data...")
    with open(tr_src, 'a') as txt_in, open(tr_tgt, 'a') as txt_out:
        for i in tqdm(range(args.train_size)):
            code_str, tgt_str = get_data_pair(args.code_length)
            # input_seq = ' '.join(code_str.split())
            input_seq = code_str
            output_seq = tgt_str
            # visualize(code_str, tgt_str)
            # print(input_seq)
            # print(tgt_str)
            if i != args.train_size - 1:
                txt_in.write(input_seq + '\n')
                txt_out.write(output_seq + '\n')

    # valid
    print("done.")
    print("Generating valid data...")
    with open(valid_src, 'a') as txt_in, open(valid_tgt, 'a') as txt_out:
        for i in tqdm(range(args.valid_size)):
            code_str, tgt_str = get_data_pair(args.code_length)
            # input_seq = ' '.join(code_str.split())
            input_seq = code_str
            output_seq = tgt_str
            # visualize(code_str, tgt_str)
            # print(input_seq)
            # print(tgt_str)

            if i != args.valid_size - 1:
                txt_in.write(input_seq + '\n')
                txt_out.write(output_seq + '\n')

    # test
    print("done.")
    print("Generating test data...")
    with open(test_src, 'a') as txt_in, open(test_tgt, 'a') as txt_out:
        for i in tqdm(range(args.test_size)):
            code_str, tgt_str = get_data_pair(args.code_length)
            input_seq = code_str
            # input_seq = ' '.join(code_str.split())
            output_seq = tgt_str
            # visualize(code_str, tgt_str)
            # print(input_seq)
            # print(tgt_str)
            if i != args.test_size - 1:
                txt_in.write(input_seq + '\n')
                txt_out.write(output_seq + '\n')
