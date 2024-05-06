import json
import os

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TRAINSET_RATIO, label_dict


# 将txt/json数据用dataframe读出
def read_data(file, enc):
    data_list = []
    with open(file, 'r', encoding=enc) as f:
        while True:
            data = f.readline()
            if not data:
                break
            data = json.loads(data)
            sentence = data['originalText']
            entities = data['entities']
            data_list.append([sentence, entities])
    return data_list


def BIOtag(sentence, entities):
    label = ['O'] * len(sentence)
    for entity in entities:
        start_idx = entity['start_pos']
        end_idx = entity['end_pos']
        type_cn = entity['label_type']
        type = label_dict[type_cn]
        # 为实体设置BIO格式标签
        label[start_idx] = 'B-' + type
        for i in range(start_idx + 1, end_idx):
            label[i] = 'I-' + type
    return label


def process(raw_data):
    processed_data = []
    for data in raw_data:
        sentence = data[0]
        entities = data[1]
        label = BIOtag(sentence, entities)
        assert len(sentence) == len(label)
        processed_data.append([list(sentence), label])
    return processed_data


def savefile(file, datas):
    with open(file, 'w',encoding='utf-8') as f:
        for data in datas:
            size = len(data[0])
            for i in range(size):
                f.write(data[0][i])
                f.write('\t')
                f.write(data[1][i])
                f.write('\n')


if __name__ == '__main__':
    # raw_data (from txt/json)
    raw_data_train_part1 = read_data(RAW_DATA_PATH + 'subtask1_training_part1.txt', 'utf-8-sig')
    raw_data_train_part2 = read_data(RAW_DATA_PATH + 'subtask1_training_part2.txt', 'utf-8-sig')
    raw_data_train = raw_data_train_part1 + raw_data_train_part2
    raw_data_test = read_data(RAW_DATA_PATH + 'subtask1_test_set_with_answer.json', 'utf-8')
    # processed_data (convert to BIO tag)
    data_train = process(raw_data_train)
    data_test = process(raw_data_test)
    # split
    num = len(data_train)
    train_data = data_train[:int(num * TRAINSET_RATIO)]
    val_data = data_train[int(num * TRAINSET_RATIO):]
    test_data = data_test
    # save
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)
    savefile(PROCESSED_DATA_PATH + "train_data.txt", val_data)
    savefile(PROCESSED_DATA_PATH + "val_data.txt", val_data)
    savefile(PROCESSED_DATA_PATH + "test_data.txt", test_data)

    print("preprocess done!")
