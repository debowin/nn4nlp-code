import argparse
import math
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from bleu import bleu, bleu_stats
from torch import nn
from torch.autograd import Variable
from torchtext.datasets import Multi30k #German to English dataset
from torchtext.data import Field, BucketIterator


def get_datasets(batch_size=128):

    # define the tokenizers
    def tokenize_de(text):
        return text.split()

    def tokenize_en(text):
        return text.split()

    # Create the pytext's Field
    Source = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
    Target = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

    # Splits the data in Train, Test and Validation data
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(Source, Target), root = 'data')

    # Build the vocabulary for both the language
    Source.build_vocab(train_data, min_freq=3)
    Target.build_vocab(train_data, min_freq=3)

    # Create the Iterator using builtin Bucketing
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
                                                                          batch_size=batch_size,
                                                                          sort_within_batch=True,
                                                                          sort_key=lambda x: len(x.src),
                                                                          device=torch.device("cuda") if USE_CUDA else torch.device("cpu"))
    return train_iterator, valid_iterator, test_iterator, Source, Target


# Model parameters
NLAYERS = 2
DROPOUT = 0.3
EMBED_SIZE = 256
HIDDEN_SIZE = 1024
BATCH_SIZE = 512

USE_CUDA = torch.cuda.is_available()

train_it, valid_it, test_it, src_vocab, trg_vocab = get_datasets(BATCH_SIZE)
pad_src = src_vocab.pad_token
pad_trg = trg_vocab.pad_token

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
            num_embeddings=nwords, embedding_dim=emb_size
        )
        torch.nn.init.uniform_(self.src_embedding.weight, -0.25, 0.25)

        self.enc_lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=nlayers,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_words):
        # encode
        src_embedding = self.dropout(
            self.src_embedding(src_words)
        )  # seqln * bsz * embsz
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
        torch.nn.init.uniform_(self.trg_embedding.weight, -0.25, 0.25)

        self.dec_lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=nlayers,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

        self.projection_layer = torch.nn.Linear(hidden_size, nwords, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)

    def forward(self, trg_words, in_hidden, in_cell):
        # decode
        trg_embedding = self.dropout(
            self.trg_embedding(trg_words.unsqueeze(0))
        )  # 1 * bsz * embsz
        outputs, (out_hidden, out_cell) = self.dec_lstm(
            trg_embedding, (in_hidden, in_cell)
        )  # 1 * bsz * hidsz, nlayers * bsz * hidsz, nlayers * bsz * hidsz
        logits = self.projection_layer(outputs.squeeze(0))  # (bsz) * nwords_trg
        return (
            logits,
            out_hidden,
            out_cell,
        )  # (bsz) * nwords_trg, bsz * hidsz, bsz * hidsz


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
        self.encoder = Encoder(nwords_src, emb_size, hidden_size, nlayers, dropout)
        self.decoder = Decoder(nwords_trg, emb_size, hidden_size, nlayers, dropout)
        self.nwords_trg = nwords_trg

    def forward(self, src_words, trg_words, tfr):
        max_trg_len, bsz = trg_words.shape
        outputs = torch.zeros(max_trg_len, bsz, self.nwords_trg)  # seqln * bsz * nwords
        if USE_CUDA:
            outputs = outputs.to(torch.device("cuda"))
        prev_word = trg_words[0]  # bsz
        prev_hidden, prev_cell = self.encoder(src_words)
        for i in range(1, max_trg_len):
            logits, prev_hidden, prev_cell = self.decoder(
                prev_word, prev_hidden, prev_cell
            )  # bsz * nwords_trg
            outputs[i] = logits
            best_guess = logits.argmax(1)  # bsz
            prev_word = trg_words[i] if random.random() < tfr else best_guess
        return outputs


def calc_loss(batch, tfr=0.5):
    # Transduce all batch elements with an LSTM
    src = batch.src
    trg = batch.trg

    logits = model(src, trg, tfr)  # seqln * bsz * nwords

    nwords_trg = logits.shape[-1]

    loss = criterion(logits[1:].view(-1, nwords_trg), trg[1:].reshape(-1))

    return loss


def generate(src):
    src = src.unsqueeze(1)
    trg = [[trg_vocab.vocab.stoi["<sos>"] for _ in range(MAX_SENT_SIZE)]]
    trg = torch.LongTensor(trg).transpose(1, 0)
    if USE_CUDA:
        trg = trg.to(torch.device("cuda"))
    # feed the previous word into the lstm, calculate the most likely word, add it to the sentence
    logits = model(src, trg, 0.0) # (seqln) * nwords
    preds = logits[1:].view(MAX_SENT_SIZE-1, -1).argmax(1).tolist()
    # cut the list at the first eos
    return preds[:preds.index(trg_vocab.vocab.stoi["<eos>"])]


model = Seq2SeqModel(
    nwords_src=len(src_vocab.vocab),
    nwords_trg=len(trg_vocab.vocab),
    emb_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE,
    nlayers=NLAYERS,
    dropout=DROPOUT,
)
if USE_CUDA:
    print("Using GPU")
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
pad_trg_idx = trg_vocab.vocab.stoi[pad_trg]
criterion = nn.CrossEntropyLoss(ignore_index=pad_trg_idx)

best_dev = 1e20
for ITER in range(30):
    # Perform training
    model.train()
    train_loss = 0.0
    start_time = time.time()
    for sent_id, train_batch in enumerate(train_it):
        optimizer.zero_grad()
        my_loss = calc_loss(train_batch)
        train_loss += my_loss.item()
        my_loss.backward()
        optimizer.step()
        if (sent_id + 1) % (2000 / BATCH_SIZE) == 0:
            print("[TRAIN] --finished %r batches" % (sent_id + 1))
    print(
        "[TRAIN] iter %r: train loss=%.4f, time=%.2fs"
        % (
            ITER,
            train_loss,
            time.time() - start_time,
        )
    )
    model.eval()
    sample_batch_id = random.choice(range(len(valid_it)))
    # Evaluate on dev set
    dev_loss = 0.0
    start_time = time.time()
    for sent_id, dev_batch in enumerate(valid_it):
        if sent_id == sample_batch_id:
            sample_id = random.choice(range(len(dev_batch)))
            target_txt = list(map(lambda x: trg_vocab.vocab.itos[x], dev_batch.trg[1:, sample_id]))
            pred_txt = map(lambda x: trg_vocab.vocab.itos[x], generate(dev_batch.src[:, sample_id]))
            print("[DEV] Target:\t" + " ".join(target_txt[:target_txt.index("<eos>")]))
            print("[DEV] Pred:\t" + " ".join(pred_txt))
        my_loss = calc_loss(dev_batch, 0.0)
        dev_loss += my_loss.item()
    print(
        "[DEV] iter %r: dev loss=%.4f, time=%.2fs"
        % (
            ITER,
            dev_loss,
            time.time() - start_time,
        )
    )
    if best_dev > dev_loss:
        print("[DEV] Best model so far, saving snapshot.")
        torch.save(model, "batched_enc_dec_model.pt")
        best_dev = dev_loss

    # this is how you generate, can replace with desired sentenced to generate
model = torch.load("batched_enc_dec_model.pt")
_, _, test_it, _, _ = get_datasets(1)
sentences = []
stats = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
model.eval()
for sent in test_it:
    pred_txt = " ".join(map(lambda x: trg_vocab.vocab.itos[x], generate(sent.src[:, 0])))
    sentences.append(pred_txt)
    target_txt = list(map(lambda x: trg_vocab.vocab.itos[x], sent.trg[1:, 0]))
    stats += np.array(
        bleu_stats(
            pred_txt,
            " ".join(target_txt[:target_txt.index("<eos>")])
        )
    )
print("Corpus BLEU: %.2f" % (100 * bleu(stats)))
for sent in sentences[:10]:
    print(sent)
