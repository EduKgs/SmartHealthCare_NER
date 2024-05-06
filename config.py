import torch

# device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# dataset
TRAINSET_RATIO = 0.2
# data path
RAW_DATA_PATH = 'data/ccks2019/'
#RAW_DATA_PATH= 'data/cmeee/'
PROCESSED_DATA_PATH = 'data/ccks2019/processed_data/'
PROCESSED_DATA_PATH = 'data/cmeee/processed_data/'

SAVED_MODEL_PATH = 'saved_model/'
# model parameter
BASE_MODEL = 'ernie-health-zh'
EMBEDDING_DIM = 768
HIDDEN_DIM = 256
# train parameter
BATCH_SIZE = 8
LR = 0.001
EPOCHS = 10
NUM_HEADS = 8
# tag&label CCKS2019
label_dict = {'药物': 'DRUG',
              '解剖部位': 'BODY',
              '疾病和诊断': 'DISEASES',
              '影像检查': 'EXAMINATIONS',
              '实验室检验': 'TEST',
              '手术': 'TREATMENT'}
label_dict2 = {'DRUG': '药物',
               'BODY': '解剖部位',
               'DISEASES': '疾病和诊断',
               'EXAMINATIONS': '影像检查',
               'TEST': '实验室检验',
               'TREATMENT': '手术'}
model_tag = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-BODY', 'I-TEST', 'I-EXAMINATIONS',
             'I-TREATMENT', 'B-DRUG', 'B-TREATMENT', 'I-DISEASES', 'B-EXAMINATIONS',
             'I-BODY', 'B-TEST', 'B-DISEASES', 'I-DRUG')
tag2idx = {tag: idx for idx, tag in enumerate(model_tag)}
idx2tag = {idx: tag for idx, tag in enumerate(model_tag)}
LABELS = ['B-BODY', 'B-DISEASES', 'B-DRUG', 'B-EXAMINATIONS', 'B-TEST', 'B-TREATMENT',
          'I-BODY', 'I-DISEASES', 'I-DRUG', 'I-EXAMINATIONS', 'I-TEST', 'I-TREATMENT']


# tag&label CMeEE
# label_dict = {'疾病': 'DIS',
#               '临床表现': 'SYM',
#               '药物': 'DRU',
#               '医疗设备': 'EQU',
#               '医疗程序': 'PRO',
#               '身体': 'BOD',
#               '医学检验项目': 'ITE',
#               '微生物类': 'MIC',
#               '科室': 'DEP'
#               }
# label_dict2 = {'DIS': '疾病',
#               'SYM': '临床表现',
#               'DRU': '药物',
#               'EQU': '医疗设备',
#               'PRO': '医疗程序',
#               'BOD': '身体',
#               'ITE': '医学检验项目',
#               'MIC': '微生物类',
#               'DEP': '科室'}
# model_tag = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-DIS', 'I-DIS', 'B-SYM','I-SYM','B-DRU','I-DRU','B-EQU',
#              'I-EQU','I-PRO', 'B-PRO', 'B-BOD', 'I-BOD', 'B-ITE','I-DEP', 'I-ITE', 'B-MIC', 'B-DEP', 'I-MIC')
# tag2idx = {tag: idx for idx, tag in enumerate(model_tag)}
# idx2tag = {idx: tag for idx, tag in enumerate(model_tag)}
# LABELS = ['B-DIS', 'B-SYM', 'B-DRU', 'B-EQU', 'B-PRO', 'B-BOD','B-ITE','B-MIC','B-DEP',
#           'I-DIS', 'I-SYM', 'I-DRU', 'I-EQU', 'I-PRO', 'I-BOD','I-IEE','I-MIC','I-DEP']
#


