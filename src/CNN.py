import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import os
from collections import Counter
import numpy as np
from torch.utils.data import DataLoader, Dataset
import re
import random
import matplotlib.pyplot as plt
import argparse


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





def read_files(class_, split="train"):
    texts = []
    for text_file in os.listdir(f"data/{split}/{class_}"):
        with open(f"data/{split}/{class_}/{text_file}", "r") as file:
            text = file.read()
        texts.append(text)

    return texts



def get_most_common_words(positive_sentences, negative_sentences):
    all_sentences = [" ".join(t for t in positive_sentences  + negative_sentences)]

    all_sentences = re.sub(r'[^\w\s]', '', all_sentences[0])


    word_freq = Counter(all_sentences.split())

    most_common = word_freq.most_common(9997)

    common_tokens = list(map(lambda x: x[0], most_common))

    common_tokens.append("UNK")
    common_tokens.append("PAD")
    return common_tokens, len(word_freq)


def get_weights_and_tokens():

    with open("data/all.review.vec.txt", "r") as file:
        weights_file = file.readlines()

    weights = []
    tokens = []
    for line in weights_file[1:]:   # exclude the first line 50560, 100)
        line = line.split()
        tokens.append(line[0])
        weights.append(
            np.array(line[1:]).astype(float)
            )

    return weights, tokens



weights, tokens = get_weights_and_tokens()

token_weights = {k: v for k, v in zip(tokens, weights)}

positive_sentences = read_files("positive")
negative_sentences = read_files("negative")



most_commmon, no_of_unique_words = get_most_common_words(positive_sentences, negative_sentences)

print(f"\nThe total number of unique words are {no_of_unique_words}\n")

number_of_training_folders = os.listdir("data/train/negative") + os.listdir("data/train/positive")

print(f"The total number of training examples in T are {len(number_of_training_folders)}\n")

ratio_pos_to_neg = len(os.listdir("data/train/positive")) / len(os.listdir("data/train/negative"))

print(f"the ratio of positive examples to negative examples in T is {ratio_pos_to_neg}\n")

max_len_doc = 0.0
sum_len_doc = 0.0
for doc in positive_sentences + negative_sentences:
    doc = re.sub(r'[^\w\s]', '', doc)
    doc = doc.split()
    if len(doc)>max_len_doc:
        max_len_doc = len(doc)
    sum_len_doc += len(doc)


print(f"The average length of document in T is {sum_len_doc/ len(number_of_training_folders)}\n")

print(f"The max length of document in T is {max_len_doc}\n")


print("="*70, "\n")


"""
make new matrix of 10000 tokens
"""


new_matrix =[]

for token in most_commmon:
    if token in token_weights:
        new_matrix.append(
            token_weights[token]
        )
    else:
        new_matrix.append(np.zeros([100]))

pretrained_embeddings = torch.from_numpy(
    np.array(new_matrix, dtype=np.float32)
)

TOKENS_DICT = {
    k: v for v,k in enumerate(most_commmon)
}



def prepare_sentence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else to_ix["UNK"] for w in seq]
    # idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)



class TrainDataset(Dataset):
    def __init__(self):
        super(TrainDataset, self).__init__()

        positive_sentences = read_files("positive")
        negative_sentences = read_files("negative")

        self.all_sentences = positive_sentences + negative_sentences

        self.all_labels = [0]*len(positive_sentences) + [1]*len(negative_sentences)
        
        self.to_ix = TOKENS_DICT

    def __len__(self):
        return len(self.all_sentences)
    
    def __getitem__(self, index):
        
        sentence = self.all_sentences[index].split()

        if len(sentence) < 100:
            sentence = sentence + (100 - len(sentence))*["PAD"]

        else:
            sentence = sentence[:100]  

        sentence = prepare_sentence(sentence, self.to_ix)

          # max len to 100



        label = self.all_labels[index]

        return  sentence, label      


class TestDataset(Dataset):
    def __init__(self):
        super(TestDataset, self).__init__()

        positive_sentences = read_files("positive", split="test")
        negative_sentences = read_files("negative", split="test")

        self.all_sentences = positive_sentences + negative_sentences

        self.all_labels = [0]*len(positive_sentences) + [1]*len(negative_sentences)
        
        self.to_ix = TOKENS_DICT

    def __len__(self):
        return len(self.all_sentences)
    
    def __getitem__(self, index):
        
        sentence = self.all_sentences[index].split()

        if len(sentence) < 100:
            sentence = sentence + (100 - len(sentence))*["PAD"]

        else:
            sentence = sentence[:100]  

        sentence = prepare_sentence(sentence, self.to_ix)

          # max len to 100


        label = self.all_labels[index]

        return  sentence, label         


class CNNClassifier(nn.Module):
    def __init__(self, load_embeddings):
        super(CNNClassifier, self).__init__()

        if load_embeddings:
            self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        else:

            self.word_embeddings = nn.Embedding(10000, embedding_dim=100, padding_idx=9999)

        self.cnn = nn.Conv1d(100, 100, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(100, 100)

        self.linear2 = nn.Linear(100,2)

    def forward(self, x):
        x = self.word_embeddings(x)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        # x = F.relu(x)

        x = torch.mean(x, dim=1)
        
        x = self.linear1(x)
    
        x = self.linear2(x)

        return x
    


def train(model, dataloader, criterion, optimizer):
    model.train()

    train_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for text, label in tqdm(dataloader):
        model.zero_grad()

        output = model(text)

        loss = criterion(output, label)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(output, 1)
        correct_predictions += (predicted == label).sum().item()
        total_predictions += label.size(0)

    accuracy = correct_predictions / total_predictions
    return train_loss / len(dataloader), accuracy


def test(model, dataloader, criterion):
    model.eval()

    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for text, label in tqdm(dataloader):
            output = model(text)

            loss = criterion(output, label)
            val_loss += loss.item()

           
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == label).sum().item()
            total_predictions += label.size(0)

    accuracy = correct_predictions / total_predictions
    return val_loss / len(dataloader), accuracy


def run(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr = 0.01, params=model.parameters())

    train_losses = []
    test_losses = []

    train_accs = []
    test_accs = []

    for i in range(7):
        train_loss, acc = train(model,
                                train_loader,
                                criterion,
                                optimizer
                                )
        
        val_loss, val_acc = test(model,
                                 test_loader,
                                 criterion)
        
        train_losses.append(train_loss)
        test_losses.append(val_loss)

        train_accs.append(acc)
        test_accs.append(val_acc)

        print(f"Epoch {i+1}---train_loss--{train_loss}---train_acc---{acc}-----test_loss---{val_loss}----test_acc---{val_acc}")



    plt.figure(figsize=(10, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy', color='green')
    plt.plot(test_accs, label='Test Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Save the plot as a file
    plt.tight_layout()
    plt.savefig('CNN-Metrics.png')

if __name__ == "__main__":

    set_seed(42)
    train_dataset = TrainDataset()
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=64,
                                  shuffle=True)
    


    test_dataset = TestDataset()
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32
    )

    parser = argparse.ArgumentParser(description='CNN Classifier')
    parser.add_argument('--load_embeddings', type=bool, default=False, help='Load pretrained word embeddings')
    args = parser.parse_args()

    model = CNNClassifier(load_embeddings=args.load_embeddings)

    run(model, train_dataloader, test_dataloader)



        