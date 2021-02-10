from collections import defaultdict
import math
import numpy as np
import time
import random
import torch
import torch.nn.functional as F


class WordEmbSkip(torch.nn.Module):
    def __init__(self, nwords, emb_size):
        super(WordEmbSkip, self).__init__()

        """ word embeddings """
        self.word_embedding = torch.nn.Embedding(nwords, emb_size, sparse=False)
        # initialize the weights with xavier uniform (Glorot, X. & Bengio, Y. (2010))
        torch.nn.init.xavier_uniform_(self.word_embedding.weight)
        """ context embeddings"""
        self.context_embedding = torch.nn.Embedding(nwords, emb_size, sparse=False)
        # initialize the weights with xavier uniform (Glorot, X. & Bengio, Y. (2010))
        torch.nn.init.xavier_uniform_(self.context_embedding.weight)

    # useful ref: https://arxiv.org/abs/1402.3722
    def forward(self, word_pos, context_positions, negative_sample=False):
        embed_word = self.word_embedding(word_pos) # bsz * 1 * emb_size
        embed_context = self.context_embedding(context_positions)  # bsz * n * emb_size
        score = torch.bmm(embed_context, embed_word.permute(0, 2, 1)) #score = bsz * n * 1

        # following is an example of something you can only do in a framework that allows
        # dynamic graph creation
        if negative_sample:
              score = -score
        obj = -torch.sum(F.logsigmoid(score), dim=1) # bsz * 1
        return obj

NEG_SAMPLES=3 #number of negative samples
WINDOW_SIZE=2 #length of window on each side (so N=2 gives a total window size of 5, as in t-2 t-1 t t+1 t+2)
EMB_SIZE = 128 # The size of the embedding
MB_SIZE = 64

embeddings_location = "embeddings.txt" #the file to write the word embeddings to
labels_location = "labels.txt" #the file to write the labels to

# We reuse the data reading from the language modeling class
w2i = defaultdict(lambda: len(w2i))

#word counts for negative sampling
word_counts = defaultdict(int)

S = w2i["<s>"]
UNK = w2i["<unk>"]
def read_dataset(filename):
  with open(filename, "r") as f:
    for line in f:
      line = line.strip().split(" ")
      for word in line:
        word_counts[w2i[word]] += 1
      yield [w2i[x] for x in line]


# Read in the data
train = list(read_dataset("../data/ptb/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("../data/ptb/valid.txt"))
i2w = {v: k for k, v in w2i.items()}
nwords = len(w2i)

# write words
with open(labels_location, 'w') as labels_file:
  for i in range(nwords):
    labels_file.write(i2w[i] + '\n')

# take the word counts to the 3/4, normalize
normalizing_constant = sum([v**0.75 for v in word_counts.values()])
word_probabilities = np.zeros(nwords)
for word_id in word_counts:
  word_probabilities[word_id] = word_counts[word_id]**.75/normalizing_constant

# initialize the model
model = WordEmbSkip(nwords, EMB_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

type = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    print("Using GPU.")
    type = torch.cuda.LongTensor
    model.cuda()


# Calculate the loss value for the entire sentence
def calc_sents_loss(sents):
    # add padding to the sentence equal to the size of the window
    # as we need to predict the eos as well, the future window at that point is N past it
    mb_length = sum(len(sent) for sent in sents)
    all_neg_words = np.random.choice(nwords, size=2*WINDOW_SIZE*NEG_SAMPLES*mb_length, replace=True, p=word_probabilities)

    pos_words_tensor = torch.Tensor(mb_length, 2*WINDOW_SIZE).type(type)
    neg_words_tensor = torch.Tensor(mb_length, 2*WINDOW_SIZE*NEG_SAMPLES).type(type)
    target_word_tensor = torch.Tensor(mb_length, 1).type(type)

    # Step through the sentences
    idx = 0
    for sent in sents:
        for i, word in enumerate(sent):
            pos_words = [sent[x] if x >= 0 else S for x in range(i-WINDOW_SIZE,i)] + \
                        [sent[x] if x < len(sent) else S for x in range(i+1,i+WINDOW_SIZE+1)]
            pos_words_tensor[idx] = torch.tensor(pos_words).type(type)
            neg_words = all_neg_words[idx*NEG_SAMPLES*2*WINDOW_SIZE:(idx+1)*NEG_SAMPLES*2*WINDOW_SIZE]
            neg_words_tensor[idx] = torch.tensor(neg_words).type(type)
            target_word_tensor[idx] = torch.tensor([word]).type(type)
            idx += 1

    #NOTE: technically, one should ensure that the neg words don't contain the
    #      context (i.e. positive) words, but it is very unlikely, so we can ignore that
    pos_loss = model(target_word_tensor, pos_words_tensor)
    neg_loss = model(target_word_tensor, neg_words_tensor, negative_sample=True)

    total_loss = torch.sum(pos_loss + neg_loss)

    return total_loss, mb_length

train.sort(key=lambda x: -len(x))
dev.sort(key=lambda x: -len(x))
train_order = [x * MB_SIZE for x in range(int((len(train) - 1) / MB_SIZE + 1))]
dev_order = [x * MB_SIZE for x in range(int((len(dev) - 1) / MB_SIZE + 1))]

best_loss = 1e20
for ITER in range(20):
    print("started iter %r" % ITER)
    # Perform training
    random.shuffle(train_order)
    train_words, train_loss = 0, 0.0
    start = time.time()
    model.train()
    for i, sid in enumerate(train_order):
        my_loss, mb_words = calc_sents_loss(train[sid:sid + MB_SIZE])
        train_loss += my_loss.item()
        train_words += mb_words
        # Back prop while training
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()
        if (i + 1) % int(10000/MB_SIZE) == 0:
            print("[TRAIN] --finished %r mini-batches" % (i + 1))
            train_ppl = float('inf') if train_loss / train_words > 709 else math.exp(train_loss / train_words)
            print("[TRAIN] after mini-batches %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
			i + 1, train_loss / train_words, train_ppl, time.time() - start))
    train_ppl = float('inf') if train_loss / train_words > 709 else math.exp(train_loss / train_words)
    print("[TRAIN] iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
    ITER, train_loss / train_words, train_ppl, time.time() - start))
    # Evaluate on dev set
    dev_words, dev_loss = 0, 0.0
    start = time.time()
    model.eval()
    for i, sid in enumerate(dev_order):
        my_loss, mb_words = calc_sents_loss(dev[sid:sid + MB_SIZE])
        dev_loss += my_loss.item()
        dev_words += mb_words
    dev_ppl = float('inf') if dev_loss / dev_words > 709 else math.exp(dev_loss / dev_words)
    print("[DEV] iter %r: dev loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
    ITER, dev_loss / dev_words, dev_ppl, time.time() - start))
    if dev_loss < best_loss:
        best_loss = dev_loss
        print("[DEV] best model so far, saving embedding files")
        with open(embeddings_location, 'w') as embeddings_file:
            W_w_np = model.word_embedding.weight.data.cpu().numpy()
            for i in range(nwords):
                ith_embedding = '\t'.join(map(str, W_w_np[i]))
                embeddings_file.write(ith_embedding + '\n')
