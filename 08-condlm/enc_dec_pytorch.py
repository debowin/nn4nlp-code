import time

from collections import defaultdict
import random
import math
import sys
import argparse

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

# much of the beginning is the same as the text retrieval
# format of files: each line is "word1 word2 ..." aligned line-by-line
train_src_file = "../data/parallel/train.ja"
train_trg_file = "../data/parallel/train.en"
dev_src_file = "../data/parallel/dev.ja"
dev_trg_file = "../data/parallel/dev.en"
test_src_file = "../data/parallel/test.ja"
test_trg_file = "../data/parallel/test.en"


w2i_src = defaultdict(lambda: len(w2i_src))
w2i_trg = defaultdict(lambda: len(w2i_trg))

def read(fname_src, fname_trg):
    """
    Read parallel files where each line lines up
    """
    with open(fname_src, "r") as f_src, open(fname_trg, "r") as f_trg:
        for line_src, line_trg in zip(f_src, f_trg):
            #need to append EOS tags to at least the target sentence
            sent_src = [w2i_src[x] for x in line_src.strip().split() + ['</s>']]
            sent_trg = [w2i_trg[x] for x in ['<s>'] + line_trg.strip().split() + ['</s>']]
            yield (sent_src, sent_trg)

# Read the data
train = list(read(train_src_file, train_trg_file))
unk_src = w2i_src["<unk>"]
eos_src = w2i_src['</s>']
w2i_src = defaultdict(lambda: unk_src, w2i_src)
unk_trg = w2i_trg["<unk>"]
eos_trg = w2i_trg['</s>']
sos_trg = w2i_trg['<s>']
w2i_trg = defaultdict(lambda: unk_trg, w2i_trg)
i2w_trg = {v: k for k, v in w2i_trg.items()}

nwords_src = len(w2i_src)
nwords_trg = len(w2i_trg)
dev = list(read(dev_src_file, dev_trg_file))
test = list(read(test_src_file, test_trg_file))

# Model parameters
NLAYERS = 1
EMBED_SIZE = 64
HIDDEN_SIZE = 128
DROPOUT = 0.0

USE_CUDA = torch.cuda.is_available()

# Especially in early training, the model can generate basically infinitely without generating an EOS
# have a max sent size that you end at
MAX_SENT_SIZE = 50

class EncDecModel(nn.Module):
    def __init__(self, nwords_src, nwords_trg, emb_size, hidden_size, nlayers, dropout):
        super(EncDecModel, self).__init__()

        """ layers """
        self.src_embedding = torch.nn.Embedding(
            num_embeddings=nwords_src,
            embedding_dim=emb_size
        )
        self.trg_embedding = torch.nn.Embedding(
            num_embeddings=nwords_trg,
            embedding_dim=emb_size
        )
        # uniform initialization
        torch.nn.init.uniform_(self.src_embedding.weight, -0.25, 0.25)
        torch.nn.init.uniform_(self.trg_embedding.weight, -0.25, 0.25)

        self.enc_lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=nlayers,
            dropout=dropout
        )

        self.dec_lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=nlayers,
            dropout=dropout
        )

        self.projection_layer = torch.nn.Linear(hidden_size, nwords_trg, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)

    def forward(self, src_words, trg_words):
        # encode
        src_embedding = self.src_embedding(src_words) # seqln * embsz
        _, (enc_hidden, enc_cell) = self.enc_lstm(src_embedding.view(len(src_words), 1, -1)) # hidsz
        # decode
        trg_embedding = self.trg_embedding(trg_words) # seqln * embsz
        dec_out, _ = self.dec_lstm(trg_embedding.view(len(trg_words), 1, -1), (enc_hidden, enc_cell))
        logits = self.projection_layer(dec_out.view(len(trg_words), -1)) # seqln * nwords_trg
        return logits


def calc_loss(sent):
    src = convert_to_variable(sent[0])
    trg = convert_to_variable(sent[1])
    label = convert_to_variable(sent[1][1:] + [eos_trg])

    logits = model(src, trg)
    loss = criterion(logits, label)

    return loss

# convert to PyTorch variable
def convert_to_variable(list_):
    variable = Variable(torch.LongTensor(list_))
    if USE_CUDA:
        variable = variable.cuda()
    return variable


def generate(sent):
    src = convert_to_variable(sent[0])
    trg_sent = [sos_trg]
    for i in range(MAX_SENT_SIZE):
        # feed the previous word into the lstm, calculate the most likely word, add it to the sentence
        trg = convert_to_variable(trg_sent)
        logits = model(src, trg)
        next_word_logits = logits[-1]
        distr = nn.functional.softmax(next_word_logits, dim=0).cpu().detach().numpy()
        next_word = np.argmax(distr)
        if next_word == eos_trg:
            break
        trg_sent.append(next_word)
    return trg_sent

model = EncDecModel(
    nwords_src=nwords_src,
    nwords_trg=nwords_trg,
    emb_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE,
    nlayers=NLAYERS,
    dropout=DROPOUT
)
if USE_CUDA:
    print("Using GPU")
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(reduction="sum")

best_dev = 1e20
for ITER in range(20):
  # Perform training
    random.shuffle(train)
    train_words, train_loss = 0, 0.0
    model.train()
    start = time.time()
    for sent_id, sent in enumerate(train):
        optimizer.zero_grad()
        my_loss = calc_loss(sent)
        train_loss += my_loss.item()
        train_words += len(sent[0])
        my_loss.backward()
        optimizer.step()
        if (sent_id+1) % 2000 == 0:
            print("[TRAIN] --finished %r sentences" % (sent_id+1))
    print("[TRAIN] iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" % (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))
    # Evaluate on dev set
    dev_words, dev_loss = 0, 0.0
    model.eval()
    print("[DEV] Target:\t" + " ".join(map(lambda x: i2w_trg[x], dev[0][1])))
    print("[DEV] Pred:\t" + " ".join(map(lambda x: i2w_trg[x], generate(dev[0]))))
    start = time.time()
    for sent_id, sent in enumerate(dev):
        my_loss = calc_loss(sent)
        dev_loss += my_loss.item()
        dev_words += len(sent[0])
    print("[DEV] iter %r: dev loss/word=%.4f, ppl=%.4f, time=%.2fs" % (ITER, dev_loss/dev_words, math.exp(dev_loss/dev_words), time.time()-start))
    if best_dev > dev_loss:
        print("[DEV] Best model so far, saving snapshot.")
        torch.save(model, "enc_dec_model.pt")
        best_dev = dev_loss

# this is how you generate, can replace with desired sentenced to generate
model = torch.load("enc_dec_model.pt")
sentences = []
for sent_id, sent in enumerate(test):
    translated_sent = generate(sent)
    sentences.append(translated_sent)
for sent in sentences:
    print(" ".join(map(lambda x: i2w_trg[x], sent)))
