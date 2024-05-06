import argparse
import os

from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from baseline.BBCSSE import Bert_FGM_BiLSTM_SE_CRF
from dataset import NerDataset, PadBatch
from utils import *

if __name__ == "__main__":
    best_model = None
    _best_val_loss = float("inf")
    _best_val_acc = -float("inf")

    #用于处理命令行的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, required=False, help="The running mode: train or infer?")
    parser.add_argument('--ckpt_name', type=str, required=False,
                        help="The name of the trained checkpoint. (without extension)")
    parser.add_argument('--txt', type=str, required=False)
    args = parser.parse_args()

    if args.mode == 'train':
        train_dataset = NerDataset(PROCESSED_DATA_PATH + 'train_data.txt')
        print(type(train_dataset))
        # print(train_dataset)
        val_dataset = NerDataset(PROCESSED_DATA_PATH + 'val_data.txt')
        test_dataset = NerDataset(PROCESSED_DATA_PATH + 'test_data.txt')
        print('Load Data Done.')
        #处理数据
        train_iter = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                collate_fn=PadBatch,
                                pin_memory=True
                                )

        eval_iter = DataLoader(dataset=val_dataset,
                               batch_size=BATCH_SIZE,
                               shuffle=False,
                               collate_fn=PadBatch,
                               pin_memory=True)

        test_iter = DataLoader(dataset=test_dataset,
                               batch_size=BATCH_SIZE,
                               shuffle=False,
                               collate_fn=PadBatch,
                               pin_memory=True)

        model = Bert_FGM_BiLSTM_SE_CRF(tag2idx).to(DEVICE)
        for name, param in model.named_parameters():
            print(f"Name: {name}, Size: {param.size()}")
        #帮助防止过拟合
        optimizer = AdamW(model.parameters(), lr=LR, eps=1e-6)
        # Warmup
        len_dataset = len(train_dataset)
        total_steps = (len_dataset // BATCH_SIZE) * EPOCHS if len_dataset % BATCH_SIZE == 0 else (len_dataset // BATCH_SIZE + 1) * EPOCHS
        warm_up_ratio = 0.1
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_ratio * total_steps,
                                                    num_training_steps=total_steps)

        print('Train Start ...')
        for epoch in range(1, EPOCHS + 1):
            train(epoch, model, train_iter, optimizer, scheduler, DEVICE)
            print('valid-->', end='')
            ##验证集的输出结果
            candidate_model, loss, acc, f1, precision, recall  = validate(epoch, model, eval_iter, DEVICE)
            if loss < _best_val_loss and acc > _best_val_acc:
                best_model = candidate_model
                _best_val_loss = loss
                _best_val_acc = acc
        #使用最终的指标去跑测试集的数据
        y_test, y_pred = test(best_model, test_iter, DEVICE)
        if not os.path.exists(SAVED_MODEL_PATH):
            os.makedirs(SAVED_MODEL_PATH)
        torch.save({'model': best_model.state_dict()}, SAVED_MODEL_PATH + 'best.ckpt')
        print('Train End ... Model saved')

    elif args.mode == 'infer':
        print('Start infer')
        model = Bert_FGM_BiLSTM_SE_CRF(tag2idx).to(DEVICE)
        tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
        if args.ckpt_name is not None:
            if os.path.exists(f"{SAVED_MODEL_PATH}{args.ckpt_name}.ckpt"):
                print("Loading the pre-trained checkpoint...")
                ckpt = torch.load(f"{SAVED_MODEL_PATH}/{args.ckpt_name}.ckpt", map_location=DEVICE)
                model.load_state_dict(ckpt['model'])
                sentence = ['[CLS]'] + list(args.txt) + ['[SEP]']
                infer(model, tokenizer, sentence)
            else:
                print("No such file!")
                exit()
    else:
        print("mode type error!")
