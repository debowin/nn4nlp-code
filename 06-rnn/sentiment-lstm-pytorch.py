from collections import defaultdict
import time
import random

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

torch.manual_seed(1)

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
S = w2i["<s>"]
UNK = w2i["<unk>"]


def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])


# Read in the data
train = list(read_dataset("../data/classes/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("../data/classes/test.txt"))
nwords = len(w2i)
ntags = len(t2i)
i2t = {v:k for k, v in t2i.items()}
i2w = {v:k for k, v in w2i.items()}

# Define the model
EMB_SIZE = 64
HID_SIZE = 128
NLAYERS = 2
DROPOUT = 0.2

USE_CUDA = torch.cuda.is_available()

class BiLSTMClass(nn.Module):
    def __init__(self, nwords, ntags, emb_size, hidden_size, nlayers, dropout, padding_idx):
        super(BiLSTMClass, self).__init__()

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
            bidirectional=True,
            batch_first=True
        )

        self.projection_layer = torch.nn.Linear(2*hidden_size, ntags, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)
    
    def forward(self, words, seq_lengths):
        batch_size, seq_len = words.size()
        embedding = self.embedding(words) # bsz * seqln * embsz
        embedding = nn.utils.rnn.pack_padded_sequence(embedding, batch_first=True, lengths=seq_lengths) 
        lstm_out, _ = self.lstm(embedding) # bsz * seqln * (2*hidsz)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = lstm_out.contiguous()
        logits = self.projection_layer(lstm_out.view(batch_size * seq_len, -1)) # (bsz * seqln) * ntags
        return logits.view(batch_size, seq_len, -1)


model = BiLSTMClass(
    nwords=nwords, 
    ntags=ntags, 
    emb_size=EMB_SIZE, 
    hidden_size=HID_SIZE,
    nlayers=NLAYERS, 
    dropout=DROPOUT, 
    padding_idx=S
)
if USE_CUDA:
    print("Using GPU")
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss(reduction="sum")

# convert to PyTorch variable
def convert_to_variable(list_):
    variable = Variable(torch.LongTensor(list_))
    if USE_CUDA:
        variable = variable.cuda()
    return variable

# A function to calculate scores for a batch
def calc_scores(sents):
    inputs = []
    lengths = list(map(len, sents))
    # longest sentence length
    batch_seq_length = len(sents[0])
    # get the wids for each sentence
    for sent in sents:
        padded_seq = [(sent[i] if i < len(sent) else S) for i in range(batch_seq_length)]
        inputs.append(padded_seq)

    # feed word vectors into the RNN and predict the tag
    inputs = convert_to_variable(inputs) # bsz * seqln

    logits = model(inputs, lengths) # bsz * seqln * ntags

    return nn.functional.log_softmax(logits[:, -1, :], dim=1) # bsz * ntags

# Sort training sentences in descending order and count minibatches
MB_SIZE = 16
train.sort(key=lambda x: -len(x[0]))
dev.sort(key=lambda x: -len(x[0]))
train_order = [x * MB_SIZE for x in range(int((len(train) - 1) / MB_SIZE + 1))]
dev_order = [x * MB_SIZE for x in range(int((len(dev) - 1) / MB_SIZE + 1))]

# Run Training Loops
best_dev = -1
for ITER in range(20):
    # Perform training
    random.shuffle(train_order)
    model.train()
    train_loss = 0.0
    train_correct = 0.0
    start = time.time()
    for sid in train_order:
        optimizer.zero_grad()
        sents, tags = zip(*train[sid:sid+MB_SIZE])
        tags_tensor = convert_to_variable(tags)
        log_scores = calc_scores(sents)
        my_loss = criterion(log_scores, tags_tensor)
        train_loss += my_loss.item()
        my_loss.backward()
        optimizer.step()
        log_scores = log_scores.cpu().detach().numpy()
        predicts = np.argmax(log_scores, axis=1)
        train_correct += np.sum(predicts==tags)
    print("[TRAIN] iter %r: acc=%.2f%%, loss/sent=%.2f, time=%.2fs" % 
        (
            ITER, 
            train_correct * 100 / len(train), 
            train_loss / len(train),
            time.time() - start
        )
    )
    # Perform testing
    dev_correct = 0.0
    start = time.time()
    model.eval()
    for sid in dev_order:
        sents, tags = zip(*dev[sid:sid+MB_SIZE])
        log_scores = calc_scores(sents).cpu().detach().numpy()
        predicts = np.argmax(log_scores, axis=1)
        dev_correct += np.sum(predicts==tags)
    print("[DEV] iter %r: acc=%.2f%%, time=%.2fs" % 
        (
            ITER, 
            dev_correct * 100 / len(dev),
            time.time() - start
        )
    )
    if best_dev < dev_correct:
        print("[DEV] Best model so far, saving snapshot.")
        torch.save(model, "model.pt")
        best_dev = dev_correct
# Sample Batch Inference on Best Model
IB_SIZE = 7
model = torch.load("model.pt")
sid = random.choice(dev_order)
sents, tags = zip(*dev[sid:sid+IB_SIZE])
log_scores = calc_scores(sents).cpu().detach().numpy()
predicts = np.argmax(log_scores, axis=1)
print(f"SID: Sentence\tPrediction(Score)\tActual")
for i, sent in enumerate(sents):
    sent = " ".join([i2w[x] for x in sent])
    print(f"{i+1}: {sent}\t{i2t[predicts[i]]}({np.exp(max(log_scores[i])):.2f})\t{i2t[tags[i]]}")
