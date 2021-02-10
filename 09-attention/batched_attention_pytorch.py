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
from plot_attention import plot_attention


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
NLAYERS = 1
DROPOUT = 0.3
EMBED_SIZE = 512
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
        # torch.nn.init.uniform_(self.src_embedding.weight, -0.25, 0.25)

        self.enc_lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=nlayers,
            dropout=dropout,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_words):
        # encode
        src_embedding = self.dropout(
            self.src_embedding(src_words)
        )  # seqln * bsz * embsz
        outputs, (enc_hidden, enc_cell) = self.enc_lstm(src_embedding)
        # separate and concat fwd and bwd directions
        enc_hidden = torch.cat(
            (
                enc_hidden[0:enc_hidden.shape[0]:2],  # fwd
                enc_hidden[1:enc_hidden.shape[0]:2]  # bwd
            ),
            dim=2
        )  # nlayers * bsz * 2*hidsz
        enc_cell = torch.cat(
            (
                enc_cell[0:enc_cell.shape[0]:2],  # fwd
                enc_cell[1:enc_cell.shape[0]:2]  # bwd
            ),
            dim=2
        )  # nlayers * bsz * 2*hidsz

        return outputs, enc_hidden, enc_cell


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
            input_size=emb_size+2*hidden_size,
            hidden_size=hidden_size,
            num_layers=nlayers,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

        self.projection_layer = nn.Linear(hidden_size, nwords, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)

    def forward(self, trg_words, in_hidden, in_cell, context_vector):
        # decode
        trg_embedding = self.dropout(
            self.trg_embedding(trg_words.unsqueeze(0))
        )  # 1 * bsz * embsz
        trg_input = torch.cat(
            (
                trg_embedding,
                context_vector
            ),
            dim=2
        ) # 1 * bsz * (embsz + 2*hidsz)
        outputs, (out_hidden, out_cell) = self.dec_lstm(
            trg_input, (in_hidden, in_cell)
        )  # 1 * bsz * hidsz, nlayers * bsz * hidsz, nlayers * bsz * hidsz
        logits = self.dropout(self.projection_layer(outputs.squeeze(0)))  # (bsz) * nwords_trg
        return (
            logits,
            out_hidden,
            out_cell,
        )  # (bsz) * nwords_trg, nlayers * bsz * hidsz, nlayers * bsz * hidsz


class AttentiveSeq2SeqModel(nn.Module):
    def __init__(
        self,
        nwords_src,
        nwords_trg,
        emb_size,
        hidden_size,
        nlayers,
        dropout,
    ):
        super(AttentiveSeq2SeqModel, self).__init__()
        self.nwords_trg = nwords_trg
        self.dropout = nn.Dropout(dropout)
        # components
        self.encoder = Encoder(nwords_src, emb_size, hidden_size, nlayers, dropout)
        self.decoder = Decoder(nwords_trg, emb_size, hidden_size, nlayers, dropout)

        # for bidirectional encoder to unidirectional decoder.
        self.hidden_bridge_layer = nn.Linear(2*hidden_size, hidden_size, bias=True)
        self.cell_bridge_layer = nn.Linear(2*hidden_size, hidden_size, bias=True)

        torch.nn.init.xavier_uniform_(self.hidden_bridge_layer.weight)
        torch.nn.init.xavier_uniform_(self.cell_bridge_layer.weight)

        # for attention
        self.attn_combine = nn.Linear(2*hidden_size, hidden_size, bias=True)
        self.attn_energy = nn.Linear(hidden_size, 1, bias=True)

        torch.nn.init.xavier_uniform_(self.attn_combine.weight)
        torch.nn.init.xavier_uniform_(self.attn_energy.weight)

    def forward(self, src_words, trg_words, tfr):
        max_trg_len, bsz = trg_words.shape
        outputs = torch.zeros(max_trg_len, bsz, self.nwords_trg)  # trgln * bsz * nwords
        if USE_CUDA:
            outputs = outputs.to(torch.device("cuda"))
        prev_word = trg_words[0]  # bsz
        enc_outputs, enc_hidden, enc_cell = self.encoder(src_words)  # srcln * bsz * hidsz*2
        prev_hidden, prev_cell = (
            torch.relu(self.hidden_bridge_layer(enc_hidden)),
            torch.relu(self.cell_bridge_layer(enc_cell))
        )
        attn_matrix = torch.zeros(enc_outputs.shape[0], bsz, max_trg_len) # seqln * bsz * trgln
        for i in range(1, max_trg_len):
            # do attention stuff
            prev_decoder_state = torch.cat(
                (
                    # only take the last layer i.e 1 * bsz * hidsz
                    prev_hidden[-1].unsqueeze(0),
                    prev_cell[-1].unsqueeze(0)
                ),
                dim=2
            ) # 1 * bsz * hidsz
            attn_value = self.attn_combine(
                enc_outputs + prev_decoder_state
            ) # srcln * bsz * hidsz
            attn_score = self.attn_energy(
                torch.tanh(
                    attn_value
                )
            )  # seqln * bsz * 1
            attn_weights = torch.softmax(attn_score, dim=0) # seqln * bsz * 1
            context_vector = attn_weights * enc_outputs # seqln * bsz * 2*hidsz
            context_vector = torch.sum(
                context_vector,
                dim=0
            ).unsqueeze(0) # 1 * bsz * 2*hidsz

            attn_matrix[:,:,i] = attn_weights.squeeze(2)
            logits, prev_hidden, prev_cell = self.decoder(
                prev_word, prev_hidden, prev_cell, context_vector
            )  # bsz * nwords_trg
            outputs[i] = logits
            best_guess = logits.argmax(1)  # bsz
            prev_word = trg_words[i] if random.random() < tfr else best_guess
        return outputs, attn_matrix[:, :, 1:]


def calc_loss(batch, tfr=0.5):
    # Transduce all batch elements with an LSTM
    src = batch.src
    trg = batch.trg

    logits, _ = model(src, trg, tfr)  # seqln * bsz * nwords

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
    logits, attn_matrix = model(src, trg, 0.0) # (seqln) * nwords
    preds = logits[1:].view(MAX_SENT_SIZE-1, -1).argmax(1).tolist()
    # cut the list at the first eos
    return (
        preds[:preds.index(trg_vocab.vocab.stoi["<eos>"])],
        attn_matrix.squeeze()[:,:preds.index(trg_vocab.vocab.stoi["<eos>"])].cpu().detach().numpy()
    )


model = AttentiveSeq2SeqModel(
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

# best_dev = 1e20
# for ITER in range(30):
#     # Perform training
#     model.train()
#     train_loss = 0.0
#     start_time = time.time()
#     for sent_id, train_batch in enumerate(train_it):
#         optimizer.zero_grad()
#         my_loss = calc_loss(train_batch)
#         train_loss += my_loss.item()
#         my_loss.backward()
#         optimizer.step()
#         if (sent_id + 1) % (2000 / BATCH_SIZE) == 0:
#             print("[TRAIN] --finished %r batches" % (sent_id + 1))
#     print(
#         "[TRAIN] iter %r: train loss=%.4f, time=%.2fs"
#         % (
#             ITER,
#             train_loss,
#             time.time() - start_time,
#         )
#     )
#     model.eval()
#     sample_batch_id = random.choice(range(len(valid_it)))
#     # Evaluate on dev set
#     dev_loss = 0.0
#     start_time = time.time()
#     for sent_id, dev_batch in enumerate(valid_it):
#         if sent_id == sample_batch_id:
#             sample_id = random.choice(range(len(dev_batch)))
#             target_txt = list(map(lambda x: trg_vocab.vocab.itos[x], dev_batch.trg[1:, sample_id]))
#             pred_txt = map(lambda x: trg_vocab.vocab.itos[x], generate(dev_batch.src[:, sample_id])[0])
#             print("[DEV] Target:\t" + " ".join(target_txt[:target_txt.index("<eos>")]))
#             print("[DEV] Pred:\t" + " ".join(pred_txt))
#         my_loss = calc_loss(dev_batch, 0.0)
#         dev_loss += my_loss.item()
#     print(
#         "[DEV] iter %r: dev loss=%.4f, time=%.2fs"
#         % (
#             ITER,
#             dev_loss,
#             time.time() - start_time,
#         )
#     )
#     if best_dev > dev_loss:
#         print("[DEV] Best model so far, saving snapshot.")
#         torch.save(model, "batched_enc_dec_model.pt")
#         best_dev = dev_loss

# this is how you generate, can replace with desired sentenced to generate
model = torch.load("batched_enc_dec_model.pt")
_, _, test_it, _, _ = get_datasets(1)
sentences = []
stats = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
model.eval()
sample_batch_id = random.choice(range(len(test_it)))
for sent_id, sent in enumerate(test_it):
    pred, attn_matrix = generate(sent.src[:, 0])
    pred_txt = " ".join(map(lambda x: trg_vocab.vocab.itos[x], pred))
    target = list(map(lambda x: trg_vocab.vocab.itos[x], sent.trg[1:, 0]))
    target_txt = " ".join(target[:target.index("<eos>")])
    stats += np.array(
        bleu_stats(pred_txt, target_txt)
    )
    sentences.append([pred_txt, target_txt])
    if sample_batch_id == sent_id:
        #now let's visualize it's attention
        plot_attention(
            [src_vocab.vocab.itos[x] for x in sent.src[:, 0]],
            [trg_vocab.vocab.itos[x] for x in pred],
            attn_matrix,
            'attention_matrix.png')
print("Corpus BLEU: %.2f" % (100 * bleu(stats)))
for pred, target in sentences[:10]:
    print("%s => %s" % (pred, target))
