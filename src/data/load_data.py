import torch

from src.layers.aux_functions import subsequent_mask

# TODO: Remove global values (?)
global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class Batch:
    "Object for holding a batch of data with mask during training"
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.shape[-1]) # torch.tensor(subsequent_mask(tgt.shape[-1]).type_as(tgt_mask.data), requires_grad=True)


# NOTE: First example - Simple copy-task. Given a random set of input symbols from a small vocabulary, the goal is to generate back those same symbols.
def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task"
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch, 10))
        data[:, 0] = 1  # TODO Why?
        src = torch.clone(data)
        tgt = torch.clone(data)
        yield Batch(src, tgt, 0)