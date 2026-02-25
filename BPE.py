from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os

def read_corpus(file_path):
    print("Loading file:", file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


VOCAB_SIZE = 16000
special_tokens = ["<S>", "</S>", "<TRANS>", "<UNK>", "<PAD>"]

tokenizer = Tokenizer(models.BPE(byte_fallback=True))

tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

source_corpus = read_corpus("en_bg_data/train.bg")
target_corpus = read_corpus("en_bg_data/train.en")

joint_corpus = [f"{s}{t}" for s, t in zip(source_corpus, target_corpus)]

tokenizer.train_from_iterator(joint_corpus, trainer)

os.makedirs("BPE_TOKENIZER", exist_ok=True)
tokenizer.model.save("BPE_TOKENIZER")
