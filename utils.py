#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
##########################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import sys
import os
import random
import nltk
from nltk.translate.bleu_score import corpus_bleu
import pickle

from tokenizers import Tokenizer, trainers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

nltk.download("punkt")


class progressBar:
    def __init__(self, barWidth=50):
        self.barWidth = barWidth
        self.period = None

    def start(self, count):
        self.item = 0
        self.period = int(count / self.barWidth)
        sys.stdout.write("[" + (" " * self.barWidth) + "]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth + 1))

    def tick(self):
        if self.item > 0 and self.item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()
        self.item += 1

    def stop(self):
        sys.stdout.write("]\n")


def readCorpus(fileName):
    print("Loading file:", fileName)
    with open(fileName, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def train_tokenizer(corpus, specialTokens, vocab_size):
    tokenizer = Tokenizer(BPE(byte_fallback=True))

    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=specialTokens)

    tokenizer.train_from_iterator(corpus, trainer)

    os.makedirs("BPE_TOKENIZER", exist_ok=True)
    tokenizer.model.save("BPE_TOKENIZER")

    return tokenizer


def prepareData(
    sourceFileName,
    targetFileName,
    sourceDevFileName,
    targetDevFileName,
    startToken,
    endToken,
    unkToken,
    padToken,
    transToken,
    vocab_size,
):
    sourceCorpus = readCorpus(sourceFileName)
    targetCorpus = readCorpus(targetFileName)

    special_tokens = [startToken, endToken, unkToken, padToken, transToken]
    tokenizer = train_tokenizer(sourceCorpus + targetCorpus, special_tokens, vocab_size)

    def format_line(s, t):
        return f"{startToken} {s} {transToken} {t} {endToken}"

    trainCorpus = [format_line(s, t) for (s, t) in zip(sourceCorpus, targetCorpus)]

    sourceDev = readCorpus(sourceDevFileName)
    targetDev = readCorpus(targetDevFileName)

    devCorpus = [format_line(s, t) for (s, t) in zip(sourceDev, targetDev)]

    print("Corpus loading completed.")

    print("Encoding corpus...")
    # .ids extracts the numerical list from the Encoding object
    train_encoded = [tokenizer.encode(s).ids for s in trainCorpus]
    dev_encoded = [tokenizer.encode(s).ids for s in devCorpus]

    return train_encoded, dev_encoded, tokenizer.get_vocab()
