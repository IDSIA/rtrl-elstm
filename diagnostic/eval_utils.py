# Utils
import torch


def compute_accuracy(model, data_iterator, loss_fn, no_print_idx, pad_value=-1,
                     show_example=False, only_nbatch=-1):
    """Compute accuracies and loss.

    :param str, split_name: for printing the accuracy with the split name.
    :param bool, show_example: if True, print some decoding output examples.
    :param int, only_nbatch: Only use given number of batches. If -1, use all
      data (default).
    returns loss, accucary char-level accuracy, print accuracy
    """
    model.eval()

    total_loss = 0.0
    corr = 0
    corr_char = 0
    corr_print = 0

    step = 0
    total_num_seqs = 0
    total_char = 0
    total_print = 0

    for idx, batch in enumerate(data_iterator):
        step += 1
        src, tgt = batch
        logits = model(src)
        target = tgt  # (B, len)

        # to compute accuracy
        output = torch.argmax(logits, dim=-1).squeeze()

        # compute loss
        logits = logits.contiguous().view(-1, logits.shape[-1])
        labels = tgt.view(-1)
        loss = loss_fn(logits, labels)
        total_loss += loss

        # sequence level accuracy
        seq_match = (torch.eq(target, output) | (target == pad_value)
                     ).all(1).sum().item()
        corr += seq_match
        total_num_seqs += src.size()[0]

        # padded part should not be counted as correct
        char_match = torch.logical_and(
          torch.logical_and(torch.eq(target, output), target != pad_value),
          target == no_print_idx).sum().item()
        corr_char += char_match
        total_char += torch.logical_and(
          target != pad_value, target == no_print_idx).sum().item()

        # Ignore non-print outputs
        print_match = torch.logical_and(
          torch.logical_and(torch.eq(target, output), target != pad_value),
          target != no_print_idx).sum().item()
        corr_print += print_match

        total_print += torch.logical_and(
          target != pad_value, target != no_print_idx).sum().item()

        if only_nbatch > 0:
            if idx > only_nbatch:
                break

    res_loss = total_loss.item() / float(step)
    acc = corr / float(total_num_seqs) * 100
    if total_char > 0:
      no_op_acc = corr_char / float(total_char) * 100
    else:
      no_op_acc = -0
    print_acc = corr_print / float(total_print) * 100

    return res_loss, acc, no_op_acc, print_acc
