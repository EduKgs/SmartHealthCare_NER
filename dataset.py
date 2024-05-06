import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from config import BASE_MODEL, tag2idx


class NerDataset(Dataset):
    # 以句号为分割符，依次从预处理的文本中读取句子
    def __init__(self, file):
        self.sentences = []
        self.labels = []
        self.tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
        self.MAX_LEN = 256 - 2

        with open(file, 'r', encoding='utf-8') as f:
            lines = [line.split('\n')[0] for line in f.readlines() if len(line.strip()) != 0]
        word_from_file = [line.split('\t')[0] for line in lines]
        tag_from_file = [line.split('\t')[1] for line in lines]

        word, tag = [], []
        for char, t in zip(word_from_file, tag_from_file):
            if char != '。' and len(word) <= self.MAX_LEN:
                word.append(char)
                tag.append(t)
            else:
                if len(word) > self.MAX_LEN:
                    self.sentences.append(['[CLS]'] + word[:self.MAX_LEN] + ['[SEP]'])
                    self.labels.append(['[CLS]'] + tag[:self.MAX_LEN] + ['[SEP]'])
                else:
                    self.sentences.append(['[CLS]'] + word + ['[SEP]'])
                    self.labels.append(['[CLS]'] + tag + ['[SEP]'])
                word, tag = [], []

    def __getitem__(self, idx):
        sentence, label = self.sentences[idx], self.labels[idx]
        sentence_ids = self.tokenizer.convert_tokens_to_ids(sentence)
        label_ids = [tag2idx[l] for l in label]
        seqlen = len(label_ids)
        return sentence_ids, label_ids, seqlen

    def __len__(self):
        return len(self.sentences)


def PadBatch(batch):
    maxlen = max([i[2] for i in batch])
    token_tensors = torch.LongTensor([i[0] + [0] * (maxlen - len(i[0])) for i in batch])
    # 可以参考config.py <PAD> 对应的是 0
    label_tensors = torch.LongTensor([i[1] + [0] * (maxlen - len(i[1])) for i in batch])
    mask = (token_tensors > 0)
    return token_tensors, label_tensors, mask
