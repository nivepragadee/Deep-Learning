Conversation opened. 2 messages. All messages read.

Skip to content
Using Gmail with screen readers
7 of 2,558
exp6 test case2
Inbox

S Arun Sethupathy
Attachments11:02 AM (1 hour ago)
#exp6 test case2 import torch import torch.nn as nn # Sample NER data (training examples) sentences = [ ["Elon", "Musk", "founded", "SpaceX"], ["Google", "is",

Kirthick Kumar <kirthickkumarnj15@gmail.com>
Attachments
11:04 AM (1 hour ago)
to jeyasuriya



---------- Forwarded message ---------
From: S Arun Sethupathy <sarunsethupathy17@gmail.com>
Date: Wed, Oct 8, 2025 at 11:02 AM
Subject: exp6 test case2
To: Kirthick Kumar <kirthickkumarnj15@gmail.com>


#exp6 test case2
import torch
import torch.nn as nn

# Sample NER data (training examples)
sentences = [
    ["Elon", "Musk", "founded", "SpaceX"],
    ["Google", "is", "in", "California"]
]
tags = [
    ["B-PER", "I-PER", "O", "B-ORG"],
    ["B-ORG", "O", "O", "B-LOC"]
]

# Create vocabularies for words and tags
word2idx = {w: i+2 for i, w in enumerate(set(w for s in sentences for w in s))}
word2idx["<PAD>"] = 0
word2idx["<UNK>"] = 1

tag2idx = {t: i+1 for i, t in enumerate(set(t for ts in tags for t in ts))}
tag2idx["<PAD>"] = 0

max_len = max(len(s) for s in sentences)

# Encode function with padding
def encode(seq, vocab, is_tag=False):
    if is_tag:
        # Tags assumed to be in vocab, no <UNK>
        idxs = [vocab[w] for w in seq]
    else:
        idxs = [vocab.get(w, vocab["<UNK>"]) for w in seq]
    return idxs + [vocab["<PAD>"]] * (max_len - len(seq))

# Prepare tensors
X = torch.tensor([encode(s, word2idx) for s in sentences])
Y = torch.tensor([encode(t, tag2idx, is_tag=True) for t in tags])

# Define the NER model (same as POS but for NER tags)
class NERModel(nn.Module):
    def __init__(self, vocab_size, tag_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tag_size)

    def forward(self, x):
        x = self.emb(x)
        x, _ = self.lstm(x)
        return self.fc(x)

model = NERModel(len(word2idx), len(tag2idx))
criterion = nn.CrossEntropyLoss(ignore_index=tag2idx["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(50):  # more epochs for better fit on small data
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output.view(-1, len(tag2idx)), Y.view(-1))
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Prediction function
def predict(sentence):
    model.eval()
    x = torch.tensor([encode(sentence, word2idx)])
    with torch.no_grad():
        preds = model(x).argmax(dim=2).squeeze().tolist()[:len(sentence)]
    idx2tag = {v: k for k, v in tag2idx.items()}
    return [idx2tag[i] for i in preds]

# Test cases
test_sentences = [
    ["Elon", "Musk", "founded", "SpaceX"],
    ["Google", "is", "in", "California"]
]

for sent in test_sentences:
    print(f"Sentence: {sent}")
    print(f"Predicted tags: {predict(sent)}")
<img width="427" height="179" alt="Screenshot 2025-10-08 110201" src="https://github.com/user-attachments/assets/b4497b77-ae63-4f09-94c1-aa46a2f098e6" />

 One attachment
  •  Scanned by Gmail
