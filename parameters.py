import torch

sourceFileName = 'en_bg_data/train.bg'
targetFileName = 'en_bg_data/train.en'
sourceDevFileName = 'en_bg_data/dev.bg'
targetDevFileName = 'en_bg_data/dev.en'

corpusFileName = 'corpusData'
wordsFileName = 'wordsData'
modelFileName = 'NMTmodel'

#device = torch.device("cuda:0")
device = torch.device("cpu")

vocab_size = 16000
layers=2
embed_size=32
hidden_size=128
n_head=2
dropout=0.1

learning_rate = 0.001
batchSize = 2
clip_grad = 5.0

maxEpochs = 1
log_every = 10
test_every = 2000
