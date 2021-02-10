import time

from collections import defaultdict
import random
import math
import sys
import argparse

import torch
from torch import nn
from torch.autograd import Variable
from bleu import bleu, bleu_stats
import numpy as np


# some of this code borrowed from Qinlan Shen's attention from the MT class last year
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


# Creates batches where all source sentences are the same length
def create_batches(sorted_dataset, max_batch_size):
    source = [x[0] for x in sorted_dataset]
    src_lengths = [len(x) for x in source]
    batches = []
    prev = src_lengths[0]
    prev_start = 0
    batch_size = 1
    for i in range(1, len(src_lengths)):
        if src_lengths[i] != prev or batch_size == max_batch_size:
            batches.append((prev_start, batch_size))
            prev = src_lengths[i]
            prev_start = i
            batch_size = 1
        else:
            batch_size += 1
    return batches


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
pad_src = w2i_src['<pad>']
w2i_src = defaultdict(lambda: unk_src, w2i_src)
unk_trg = w2i_trg["<unk>"]
eos_trg = w2i_trg['</s>']
sos_trg = w2i_trg['<s>']
pad_trg = w2i_trg['<pad>']
w2i_trg = defaultdict(lambda: unk_trg, w2i_trg)
i2w_trg = {v: k for k, v in w2i_trg.items()}

nwords_src = len(w2i_src)
nwords_trg = len(w2i_trg)
dev = list(read(dev_src_file, dev_trg_file))
test = list(read(test_src_file, test_trg_file))

# Model parameters
NLAYERS = 2
DROPOUT = 0.5
EMBED_SIZE = 64
HIDDEN_SIZE = 128
BATCH_SIZE = 32

USE_CUDA = torch.cuda.is_available()

# Especially in early training, the model can generate basically infinitely without generating an EOS
# have a max sent size that you end at
MAX_SENT_SIZE = 50

class Encoder(nn.Module):
    def __init__(
        self,
        nwords,
        emb_size,
        hidden_size,

        nlayers,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.src_embedding = torch.nn.Embedding(
            num_embeddings=nwords,
            embedding_dim=emb_size
        )
        # torch.nn.init.uniform_(self.src_embedding.weight, -0.25, 0.25)

        self.enc_lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=nlayers,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_words):
        # encode
        src_embedding = self.dropout(self.src_embedding(src_words)) # seqln * bsz * embsz
        _, (enc_hidden, enc_cell) = self.enc_lstm(src_embedding)
        return enc_hidden, enc_cell


class Decoder(nn.Module):
    def __init__(
        self,
        nwords,
        emb_size,
        hidden_size,
        nlayers,
        dropout,
    ):
        super(Decoder, self).__init__()
        self.trg_embedding = torch.nn.Embedding(
            num_embeddings=nwords,
            embedding_dim=emb_size,
        )
        # torch.nn.init.uniform_(self.trg_embedding.weight, -0.25, 0.25)

        self.dec_lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=nlayers,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

        self.projection_layer = torch.nn.Linear(hidden_size, nwords, bias=True)
        # Initializing the projection layer
        # torch.nn.init.xavier_uniform_(self.projection_layer.weight)

    def forward(self, trg_words, in_hidden, in_cell):
        # decode
        trg_embedding = self.dropout(self.trg_embedding(trg_words.unsqueeze(0))) # 1 * bsz * embsz
        outputs, (out_hidden, out_cell) = self.dec_lstm(trg_embedding, (in_hidden, in_cell))  # 1 * bsz * hidsz, nlayers * bsz * hidsz, nlayers * bsz * hidsz
        logits = self.projection_layer(outputs.squeeze(0)) # (bsz) * nwords_trg
        return logits, out_hidden, out_cell # (bsz) * nwords_trg, bsz * hidsz, bsz * hidsz


class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        nwords_src,
        nwords_trg,
        emb_size,
        hidden_size,
        nlayers,
        dropout,
    ):
        super(Seq2SeqModel, self).__init__()
        """ components """
        self.encoder = Encoder(
            nwords_src,
            emb_size,
            hidden_size,
            nlayers,
            dropout
        )
        self.decoder = Decoder(
            nwords_trg,
            emb_size,
            hidden_size,
            nlayers,
            dropout
        )

    def forward(self, src_words, trg_words, tfr):
        max_trg_len, bsz = trg_words.shape
        outputs = torch.zeros(max_trg_len, bsz, nwords_trg)  # seqln * bsz * nwords
        if USE_CUDA:
            outputs = outputs.cuda()
        prev_word = trg_words[0, :]  # bsz
        prev_hidden, prev_cell = self.encoder(src_words)
        for i in range(1, max_trg_len):
            logits, prev_hidden, prev_cell = self.decoder(
                prev_word,
                prev_hidden,
                prev_cell
            )  # bsz * nwords_trg
            outputs[i] = logits
            best_guess = logits.argmax(1)  # bsz
            prev_word = trg_words[i] if random.random() < tfr else best_guess
        return outputs


def calc_loss(sents, tfr=0.5):
    # Transduce all batch elements with an LSTM
    src_sents = [x[0] for x in sents]
    trg_sents = [x[1] for x in sents]

    trg_len = list(map(len, trg_sents))
    max_trg_len = np.max(trg_len)

    targets = []
    mb_words = 0

    # get the wids and masks for each sentence
    for trg_sent in trg_sents:
        trg = [(trg_sent[i] if i < len(trg_sent) else pad_trg) for i in range(max_trg_len)]
        targets.append(trg)
        mb_words += len(trg_sent)

    # feed word vectors into the RNN and predict the next word
    inputs = convert_to_variable(src_sents).transpose(1, 0) # seqln * bsz
    targets = convert_to_variable(targets).transpose(1, 0) # seqln * bsz

    logits = model(inputs, targets, tfr) # seqln * bsz * nwords

    loss = criterion(logits[1:].view(-1, nwords_trg), targets[1:].reshape(-1))

    return loss, mb_words

# convert to PyTorch variable
def convert_to_variable(list_):
    variable = Variable(torch.LongTensor(list_))
    if USE_CUDA:
        variable = variable.cuda()
    return variable

def generate(sent):
    src = convert_to_variable([sent[0]]).transpose(1, 0)
    trg = [sos_trg for _ in range(MAX_SENT_SIZE)]
    # feed the previous word into the lstm, calculate the most likely word, add it to the sentence
    trg = convert_to_variable([trg]).transpose(1, 0)
    logits = model(src, trg, tfr=0.0) # (seqln) * nwords
    preds = logits.view(MAX_SENT_SIZE, -1).argmax(1).tolist()
    # cut the list at the first eos
    return preds[:preds.index(eos_trg)+1]

model = Seq2SeqModel(
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
criterion = nn.CrossEntropyLoss(ignore_index=pad_trg)

best_dev = 1e20
for ITER in range(20):
    # Perform training
    model.train()
    train.sort(key=lambda t: len(t[0]), reverse=True)
    dev.sort(key=lambda t: len(t[0]), reverse=True)
    train_order = create_batches(train, BATCH_SIZE)
    dev_order = create_batches(dev, BATCH_SIZE)
    train_words, train_loss = 0, 0.0
    start_time = time.time()
    for sent_id, (start, length) in enumerate(train_order):
        optimizer.zero_grad()
        train_batch = train[start:start+length]
        my_loss, num_words = calc_loss(train_batch)
        train_loss += my_loss.item()
        train_words += num_words
        my_loss.backward()
        optimizer.step()
        if (sent_id+1) % (2000/BATCH_SIZE) == 0:
            print("[TRAIN] --finished %r sentences" % (sent_id+1))
    print(
        "[TRAIN] iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" %
        (
            ITER,
            train_loss/train_words,
            math.exp(train_loss/train_words),
            time.time()-start_time
        )
    )
    model.eval()
    sample_dev = random.choice(dev)
    print("[DEV] Target:\t" + " ".join(map(lambda x: i2w_trg[x], sample_dev[1])))
    print("[DEV] Pred:\t" + " ".join(map(lambda x: i2w_trg[x], generate(sample_dev))))
    # Evaluate on dev set
    dev_words, dev_loss = 0, 0.0
    start_time = time.time()
    for sent_id, (start, length) in enumerate(dev_order):
        dev_batch = dev[start:start+length]
        my_loss, num_words = calc_loss(dev_batch, 0.0)
        dev_loss += my_loss.item()
        dev_words += num_words
    print(
        "[DEV] iter %r: dev loss/word=%.4f, ppl=%.4f, time=%.2fs" %
        (
            ITER,
            dev_loss/dev_words,
            math.exp(dev_loss/dev_words),
            time.time()-start_time
        )
    )
    if best_dev > dev_loss:
        print("[DEV] Best model so far, saving snapshot.")
        torch.save(model, "batched_enc_dec_model.pt")
        best_dev = dev_loss

    # this is how you generate, can replace with desired sentenced to generate
model = torch.load("batched_enc_dec_model.pt")
sentences = []
stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
for sent in test:
    hyp = generate(sent)
    sentences.append(hyp)
    stats += np.array(
        bleu_stats(
            " ".join(map(lambda x: i2w_trg[x], hyp)),
            " ".join(map(lambda x: i2w_trg[x], sent[1]))
        )
    )
print("Corpus BLEU: %.2f" % (100*bleu(stats)))
for sent in sentences[:10]:
    print(" ".join(map(lambda x: i2w_trg[x], sent)))
