from __future__ import print_function
import time
from collections import defaultdict
import random
import math

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

torch.manual_seed(1)

# format of files: each line is "word1/tag2 word2/tag2 ..."
train_file = "../data/ptb/train.txt"
test_file = "../data/ptb/valid.txt"

w2i = defaultdict(lambda: len(w2i))


def read(fname):
    """
    Read a file where each line is of the form "word1 word2 ..."
    Yields lists of the form [word1, word2, ...]
    """
    with open(fname, "r") as fh:
        for line in fh:
            sent = [w2i[x] for x in line.strip().split()]
            sent.append(w2i["<s>"])
            yield sent

S = w2i["<s>"]
train = list(read(train_file))
nwords = len(w2i)
test = list(read(test_file))
i2w = {v:k for k, v in w2i.items()}
assert (nwords == len(w2i))

# Lookup parameters for word embeddings
EMBED_SIZE = 64
HIDDEN_SIZE = 128
NLAYERS = 2
DROPOUT = 0.2

USE_CUDA = torch.cuda.is_available()

class LSTMClass(nn.Module):
    def __init__(self, nwords, emb_size, hidden_size, nlayers, dropout, padding_idx):
        super(LSTMClass, self).__init__()

        """ layers """
        self.embedding = torch.nn.Embedding(
            num_embeddings=nwords,
            embedding_dim=emb_size,
            padding_idx=padding_idx
        )
        # uniform initialization
        torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)

        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=nlayers,
            dropout=dropout,
            batch_first=True
        )

        self.projection_layer = torch.nn.Linear(hidden_size, nwords, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)

    def forward(self, words, seq_lengths):
        batch_size, seq_len = words.size()
        embedding = self.embedding(words) # bsz * seqln * embsz
        embedding = nn.utils.rnn.pack_padded_sequence(embedding, batch_first=True, lengths=seq_lengths)
        lstm_out, _ = self.lstm(embedding) # bsz * seqln * hidsz
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = lstm_out.contiguous()
        logits = self.projection_layer(lstm_out.view(batch_size * seq_len, -1)) # (bsz * seqln) * nwords
        return logits


model = LSTMClass(nwords=nwords, emb_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, nlayers=NLAYERS, dropout=DROPOUT, padding_idx=S)
if USE_CUDA:
    print("Using GPU")
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(reduction="sum")

# convert to PyTorch variable
def convert_to_variable(list_):
    variable = Variable(torch.LongTensor(list_))
    if USE_CUDA:
        variable = variable.cuda()
    return variable

# Build the language model graph
def calc_lm_loss(sents):
    tot_words = 0
    inputs = []
    targets = []
    masks = []
    lengths = list(map(len, sents))
    # longest sentence length
    batch_seq_length = len(sents[0])
    # get the wids and masks for each sentence
    for sent in sents:
        padded_seq = [(sent[i] if i < len(sent) else S) for i in range(batch_seq_length)]
        inputs.append([S] + padded_seq[:-1])
        targets.append(padded_seq)
        mask = [(1 if i < len(sent) else 0) for i in range(batch_seq_length)]
        masks.append(mask)
        tot_words += sum(mask)

    # feed word vectors into the RNN and predict the next word
    inputs = convert_to_variable(inputs) # bsz * seqln
    targets = convert_to_variable(targets).view(-1) # (bsz * seqln)
    masks = convert_to_variable(masks).view(-1, 1) # (bsz * seqln) * 1

    logits = model(inputs, lengths) # (bsz * seqln) * nwords
    # zero out padded logits
    logits = logits * masks

    loss = criterion(logits, targets)

    return loss, tot_words


MAX_LEN = 100
# Generate a sentence
def generate_sents(num):
    model.eval()
    sents = [[S] for _ in range(num)]
    done = [0] * num
    while len(sents[0]) < MAX_LEN:
        in_tensor = convert_to_variable(sents) # bsz * seqln
        logits = model(in_tensor, list(map(len, sents))).view(in_tensor.shape[0], in_tensor.shape[1], -1) # bsz * seqln * nwords
        last_words_logits = logits[:, -1, :]
        distr = nn.functional.softmax(last_words_logits, dim=1).cpu().detach().numpy()
        for i in range(num):
            next_word = np.random.choice(len(last_words_logits[0]), p=distr[i])
            if done[i] or next_word == S:
                # finished generating current sentence.
                done[i] = 1
                sents[i].append(S)
                continue
            sents[i].append(next_word)
        if sum(done) == num:
            # finished generating all sentences.
            break
    return [list(filter(S.__ne__, sent)) for sent in sents]


# Sort training sentences in descending order and count minibatches
MB_SIZE = 32
train.sort(key=lambda x: -len(x))
test.sort(key=lambda x: -len(x))
train_order = [x * MB_SIZE for x in range(int((len(train) - 1) / MB_SIZE + 1))]
test_order = [x * MB_SIZE for x in range(int((len(test) - 1) / MB_SIZE + 1))]

# Perform training
best_dev = 1e20
start = time.time()
dev_time = all_tagged = this_words = this_loss = 0
for ITER in range(20):
    i = 0
    random.shuffle(train_order)
    for sid in train_order:
        i += 1
        if i % int(2000 / MB_SIZE) == 0:
            print(
                "[TRAIN] iter %r(step: %r): nll=%.2f, ppl=%.2f" %
                (
                    ITER, i, this_loss/this_words, math.exp(this_loss/this_words)
                )
            )
            all_tagged += this_words
            this_loss = this_words = 0
        if i % int(10000 / MB_SIZE) == 0:
            dev_start = time.time()
            model.eval()
            dev_loss = dev_words = 0
            for sid in test_order:
                loss_exp, mb_words = calc_lm_loss(test[sid:sid + MB_SIZE])
                dev_loss += loss_exp.item()
                dev_words += mb_words
            dev_time += time.time() - dev_start
            train_time = time.time() - start - dev_time
            print("[DEV] iter=%r, nll=%.2f, ppl=%.2f, words=%r, time=%.2f, word_per_sec=%.2f" % (
                ITER, dev_loss / dev_words,
                math.exp(dev_loss / dev_words), dev_words,
                train_time, all_tagged / train_time)
            )
            if best_dev > dev_loss:
                print("[DEV] Best model so far, saving snapshot.")
                torch.save(model, "model.pt")
                best_dev = dev_loss
        # train on the minibatch
        model.train()
        loss_exp, mb_words = calc_lm_loss(train[sid:sid + MB_SIZE])
        this_loss += loss_exp.item()
        this_words += mb_words
        optimizer.zero_grad()
        loss_exp.backward()
        optimizer.step()
    print("[TRAIN] epoch %r finished." % ITER)

# Generate a few sentences
model = torch.load("model.pt")
sents = generate_sents(5)
for i, sent in enumerate(sents):
    print(f"{i}: " + " ".join([i2w[x] for x in sent]))