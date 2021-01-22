import torch
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer


LABLE_DIC = {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'like': 4, 'sadness': 5, 'surprise': 6}


class Example:
    def __init__(self, content, label=None):
        self.content = content
        self.label = label


class Feature:
    def __init__(self, input_ids, attention_masks, token_type_ids, bce_labels=None, ce_labels=None):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.bce_labels = bce_labels
        self.ce_labels = ce_labels


def split_dataset(data_path, save_path):
    with open(data_path, mode='r') as reader:
        read = reader.readlines()
    train_contents, dev_contents = train_test_split(read, shuffle=True, random_state=626, test_size=0.1)
    with open(save_path + 'train_contents.txt', mode='w') as writer:
        for content in tqdm(train_contents, desc='Writing data'):
            writer.write(content)
    with open(save_path + 'dev_contents.txt', mode='w') as writer:
        for content in tqdm(dev_contents, desc='Writing data'):
            writer.write(content)


def example_reader(path):
    examples = []
    with open(path, mode='r') as reader:
        read = reader.readlines()
    for content in tqdm(read, desc="Reading data"):
        split_content = content.split('\n')[0].split('\t')
        example = Example(split_content[1], split_content[2])
        examples.append(example)
    return examples


def data_analysis(examples):
    dic = {'anger': 0, 'disgust': 0, 'fear': 0, 'happiness': 0, 'like': 0, 'sadness': 0, 'surprise': 0}
    for example in tqdm(examples, desc='Analysing data'):
        dic[example.label] += 1
    print(dic)


def feature_writer(path, features):
    with open(path, mode='wb') as writer:
        pickle.dump(features, writer)


def feature_reader(path):
    with open(path, mode='rb') as reader:
        feature = pickle.load(reader)
        return feature


def creat_dataloader(batch_size, *args):
    data = tuple(torch.tensor(t) for t in tqdm(args, desc='Convert to tensor'))
    input_ids, attention_masks, token_type_ids, labels = data
    dataset = TensorDataset(input_ids, attention_masks, token_type_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def encoding(examples, model_name):
    sentences, bce_labels, ce_labels = [], [], []
    tokenizer = BertTokenizer.from_pretrained(model_name)
    for example in tqdm(examples, desc='Processing data'):
        sentences.append(example.content)
        # create ce labels
        ce_labels.append(LABLE_DIC[example.label])
        # create bce labels
        bce_label = [0 for i in range(len(LABLE_DIC))]
        bce_label[LABLE_DIC[example.label]] = 1
        bce_labels.append(bce_label)
    print('===== Encoding... =====')
    encoded = tokenizer(sentences, truncation=True, padding=True)
    input_ids = encoded['input_ids']
    attention_masks = encoded['attention_mask']
    token_type_ids = encoded['token_type_ids']
    feature = Feature(input_ids, attention_masks, token_type_ids, bce_labels=bce_labels, ce_labels=ce_labels)
    return feature
