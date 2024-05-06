import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel

from config import EMBEDDING_DIM, HIDDEN_DIM, NUM_HEADS


class Bert_BiLSTM_Attention_CRF(nn.Module):
    #Embedding_dim 嵌入层维度
    #Hidden_Dim 隐藏层状态的维度
    def __init__(self, tag2idx, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,num_heads=NUM_HEADS):
        super(Bert_BiLSTM_Attention_CRF, self).__init__()
        self.tag_to_ix = tag2idx
        #存储标签的数量
        self.tagset_size = len(tag2idx)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.bert = BertModel.from_pretrained('bert-base-chinese')
        #因为是双向的LSTM模型 且包含两层  并且处理批量数据首先处理的batch_first
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim // 2,
                            num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        # 多头注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        #将Lstm映射到标签空间中
        self.linear = nn.Linear(hidden_dim, self.tagset_size)
        #使用CRF的decode方法,根据特征和掩码解码最佳的标签序列
        self.crf = CRF(self.tagset_size, batch_first=True)

    def getfeature(self, sentence,mask):
        with torch.no_grad():
            # BERT默认返回两个 last_hidden_state, pooler_output
            # last_hidden_state：输出序列每个位置的语义向量，形状为：(batch_size, sequence_length, hidden_size)
            # pooler_output：[CLS]符号对应的语义向量，经过了全连接层和tanh激活；该向量可用于下游分类任务
            outputs = self.bert(sentence, attention_mask =mask)
            embeds = outputs.last_hidden_state
        # LSTM默认返回两个 output, (h,c)
        # output:[batch_size,seq_len,hidden_dim * 2]   if birectional
        # h,c :[num_layers * 2,batch_size,hidden_dim]  if birectional
        # h 为LSTM最后一个时间步的隐层结果，c 为LSTM最后一个时间步的Cell状态
        out, _ = self.lstm(embeds)
        out = self.dropout(out)
        attn_out, _ = self.attention(out, out, out, key_padding_mask=~mask)
        attn_out = self.dropout(attn_out)
        feats = self.linear(attn_out)
        return feats
    #加入FGSM对抗训练至命名实体识别模型中进而增强模型的鲁棒性
    def forward(self, sentence, tags, mask, is_test=False):
        #sentece代表传递给模型的输入数据，通过在自然语言处理任务中代表一句话或则和一个文本序列
        feature = self.getfeature(sentence,mask)
        # training
        if not is_test:
            # return log-likelihood
            # make this value negative as our loss
            loss = -self.crf.forward(feature, tags, mask, reduction='mean')
            return loss
        # testing
        else:
            decode = self.crf.decode(feature, mask)
            return decode
