import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel
from SqueezeE import SELayer
from config import EMBEDDING_DIM, HIDDEN_DIM
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
class Bert_FGM_BiLSTM_SE_CRF(nn.Module):
    #Embedding_dim 嵌入层维度
    #Hidden_Dim 隐藏层状态的维度
    def __init__(self, tag2idx, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,reduction=16):
        super(Bert_FGM_BiLSTM_SE_CRF, self).__init__()
        self.tag_to_ix = tag2idx
        #存储标签的数量
        self.tagset_size = len(tag2idx)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.bert = BertModel.from_pretrained('ernie-health-zh')
        #因为是双向的LSTM模型 且包含两层  并且处理批量数据首先处理的batch_first
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim // 2,
                            num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        # 通道注意力机制
        self.se_layer = SELayer(hidden_dim, reduction)
        #将Lstm映射到标签空间中
        self.linear = nn.Linear(hidden_dim, self.tagset_size)
        #使用CRF的decode方法,根据特征和掩码解码最佳的标签序列
        self.crf = CRF(self.tagset_size, batch_first=True)

    def getfeature(self,sentence,mask):
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
        out = self.se_layer(out)
        attn_out = self.dropout(out)
        feats = self.linear(attn_out)
        return feats
    #加入FGSM对抗训练至命名实体识别模型中进而增强模型的鲁棒性
    #epsilon是一个超参数,代表的是扰动的大小
    def forward(self, sentence, tags, mask, epsilon=0.01, is_test=False):
        feature = self.getfeature(sentence, mask)
        fgm = FGM(self)
        if not is_test:
            # 计算原始损失
            loss = -self.crf(feature, tags, mask=mask, reduction='mean')
            loss.backward()
            fgm.attack(epsilon=epsilon, emb_name='bert.embeddings.word_embeddings')  # 使用FGM攻击

            # 计算对抗样本的损失
            perturbed_feature = self.getfeature(sentence, mask)
            perturbed_loss = -self.crf(perturbed_feature, tags, mask=mask, reduction='mean')
            perturbed_loss.backward()  # 累加对抗样本的梯度到模型中

            # 恢复模型参数
            fgm.restore(emb_name='bert.embeddings.word_embeddings')

            # 返回原始损失和对抗损失的平均值
            total_loss = (loss + perturbed_loss) / 2
            return total_loss
            # 返回原始损失和对抗损失的平均值

        else:
            # 测试时解码最佳路径
            decode = self.crf.decode(feature, mask)
            return decode
