import time
from collections import defaultdict
import random
import math
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

torch.manual_seed(1)

# format of files: each line is "word1 word2 ..."
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


train = list(read(train_file))
nwords = len(w2i)
test = list(read(test_file))
S = w2i["<s>"]
i2w = {v: k for k, v in w2i.items()}
assert (nwords == len(w2i))

# Lookup parameters for word embeddings
EMBED_SIZE = 64
HIDDEN_SIZE = 128
NLAYERS = 2
DROPOUT = 0.2
SEQ_LENGTH = 7

USE_CUDA = torch.cuda.is_available()

class LSTMClass(nn.Module):
    def __init__(self, nwords, emb_size, hidden_size, nlayers, dropout):
        super(LSTMClass, self).__init__()

        """ layers """
        self.embedding = torch.nn.Embedding(nwords, emb_size)
        # uniform initialization
        torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)

        self.lstm = nn.LSTM(
            input_size=emb_size, 
            hidden_size=hidden_size,
            num_layers=nlayers, 
            dropout=dropout
        )

        self.projection_layer = torch.nn.Linear(hidden_size, nwords, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)
    
    def forward(self, words):
        embedding = self.embedding(words) # seqln * embsz
        lstm_out, _ = self.lstm(embedding.view(len(words), 1, -1)) # seqln * bsz * hidsz
        logits = self.projection_layer(lstm_out.view(len(words), -1)) # seqln * nwords
        return logits
        

model = LSTMClass(nwords=nwords, emb_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, nlayers=NLAYERS, dropout=DROPOUT)
if USE_CUDA:
    print("Using GPU")
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(reduction="sum")

# Sort training sentences in descending order and count minibatches
train_order = list(range(len(train)))

# convert to PyTorch variable
def convert_to_variable(list_):
    variable = Variable(torch.LongTensor(list_))
    if USE_CUDA:
        variable = variable.cuda()
    return variable

# Build the language model graph
def calc_lm_loss(criterion, words):
    # get the wids and masks for each step
    tot_words = len(words)

    # feed word vectors into the RNN and predict the next word
    inputs = [S] + words[:-1]
    targets = words
    # calculate the softmax and loss
    in_tensor = convert_to_variable(inputs) # seqln
    out_tensor = convert_to_variable(targets) # seqln
    logits = model(in_tensor) # 1 * nwords
    loss = criterion(logits, out_tensor) # 1

    return loss, tot_words

MAX_LEN = 100
# Generate a sentence
def generate_sent(model):
    model.eval()
    sent = [S]
    while True:
        in_tensor = convert_to_variable(sent)
        logits = model(in_tensor)
        last_word_logits = logits[-1]
        distr = nn.functional.softmax(last_word_logits, dim=0).cpu().detach().numpy()
        next_word = np.random.choice(len(last_word_logits), p=distr)
        if next_word == S or len(sent) == MAX_LEN:
            break
        sent.append(next_word)
    return sent[1:]

# Perform training
best_dev = 1e20
start = time.time()
dev_time = all_tagged = this_words = this_loss = 0
for ITER in range(10):
    i = 0
    random.shuffle(train_order)
    for sid in train_order:
        i += 1
        if i % int(2000) == 0:
            print(
                "[TRAIN] iter %r(step: %r): nll=%.2f, ppl=%.2f" % 
                (
                    ITER, i, this_loss/this_words, math.exp(this_loss/this_words)
                )
            )
            all_tagged += this_words
            this_loss = this_words = 0
        if i % int(10000) == 0:
            dev_start = time.time()
            model.eval()
            dev_loss = dev_words = 0
            for sent in test:
                loss_exp, mb_words = calc_lm_loss(criterion, sent)
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
        loss_exp, mb_words = calc_lm_loss(criterion, train[sid])
        this_loss += loss_exp.item()
        this_words += mb_words
        optimizer.zero_grad()
        loss_exp.backward()
        optimizer.step()
    print("[TRAIN] epoch %r finished." % ITER)

# Generate a few sentences
for i in range(5):
    model = torch.load("model.pt")
    sent = generate_sent(model)
    print(f"{i}: " + " ".join([i2w[x] for x in sent]))