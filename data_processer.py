# -*- coding: utf-8 -*-
import torch
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer

LABLE_DIC = {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'like': 4, 'sadness': 5, 'surprise': 6}
TRANS_DIC = {'anger': '是愤怒吗？', 'disgust': '是恶心吗？', 'fear': '是害怕吗？', 'happiness': '是高兴吗？', 'like': '是欢喜吗？',
             'sadness': '是悲伤吗？', 'surprise': '是惊讶吗？'}
N_TRANS_DIC = {'anger': '不是愤怒吗？', 'disgust': '不是恶心吗？', 'fear': '不是害怕吗？', 'happiness': '不是高兴吗？', 'like': '不是欢喜吗？',
               'sadness': '不是悲伤吗？', 'surprise': '不是惊讶吗？'}


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


def adjust_dataset(data_path, target):
    examples = example_reader(data_path)
    target_examples = []
    adjust_examples = []
    for example in tqdm(examples):
        if example.label == target:
            target_examples.append(example)
        else:
            adjust_examples.append(example)
    assert len(target_examples) + len(adjust_examples) == len(examples)
    test_size = len(target_examples) / len(adjust_examples)
    _, save_examples = train_test_split(adjust_examples, shuffle=True, random_state=626, test_size=test_size)
    adjusted_examples = target_examples + save_examples
    return adjusted_examples


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


def qa_encoding(examples, model_name):
    sentences, bce_labels, ce_labels = [], [], []
    qusetions = []
    tokenizer = BertTokenizer.from_pretrained(model_name)
    for example in tqdm(examples, desc='Processing data'):
        for word in LABLE_DIC.keys():
            sentences.append(example.content)
            if example.label == word:
                bce_labels.append([1])
                ce_labels.append(1)
            else:
                bce_labels.append([0])
                ce_labels.append(0)
                qusetions.append(TRANS_DIC[word])
    print('===== Encoding... =====')
    encoded = tokenizer(qusetions, sentences, truncation=True, padding=True)
    input_ids = encoded['input_ids']
    attention_masks = encoded['attention_mask']
    token_type_ids = encoded['token_type_ids']
    feature = Feature(input_ids, attention_masks, token_type_ids, bce_labels=bce_labels, ce_labels=ce_labels)
    return feature


def n_qa_encoding(examples, model_name):
    sentences, bce_labels, ce_labels = [], [], []
    qusetions = []
    tokenizer = BertTokenizer.from_pretrained(model_name)
    for example in tqdm(examples, desc='Processing data'):
        for word in LABLE_DIC.keys():
            sentences.append(example.content)
            if example.label == word:
                bce_labels.append([0])
                ce_labels.append(0)
            else:
                bce_labels.append([1])
                ce_labels.append(1)
            qusetions.append(N_TRANS_DIC[word])
    print('===== Encoding... =====')
    encoded = tokenizer(qusetions, sentences, truncation=True, padding=True)
    input_ids = encoded['input_ids']
    attention_masks = encoded['attention_mask']
    token_type_ids = encoded['token_type_ids']
    feature = Feature(input_ids, attention_masks, token_type_ids, bce_labels=bce_labels, ce_labels=ce_labels)
    return feature


def n_qa_binary_encoding(examples, target, model_name):
    sentences, ce_labels = [], []
    qusetions = []
    tokenizer = BertTokenizer.from_pretrained(model_name)
    for example in tqdm(examples, desc='Processing data'):
        sentences.append(example.content)
        if example.label == target:
            ce_labels.append(0)
        else:
            ce_labels.append(1)
        qusetions.append(N_TRANS_DIC[target])
    print('===== Encoding... =====')
    encoded = tokenizer(qusetions, sentences, truncation=True, padding=True)
    input_ids = encoded['input_ids']
    attention_masks = encoded['attention_mask']
    token_type_ids = encoded['token_type_ids']
    feature = Feature(input_ids, attention_masks, token_type_ids, bce_labels=None, ce_labels=ce_labels)
    return feature


def binary_encoding(examples, target, model_name):
    sentences, bce_labels, ce_labels = [], [], []
    tokenizer = BertTokenizer.from_pretrained(model_name)
    for example in tqdm(examples, desc='Processing data'):
        sentences.append(example.content)
        if example.label == target:
            bce_labels.append([1])
            ce_labels.append(1)
        else:
            bce_labels.append([0])
            ce_labels.append(0)
    encoded = tokenizer(sentences, truncation=True, padding=True)
    input_ids = encoded['input_ids']
    attention_masks = encoded['attention_mask']
    token_type_ids = encoded['token_type_ids']
    feature = Feature(input_ids, attention_masks, token_type_ids, bce_labels=bce_labels, ce_labels=ce_labels)
    return feature


def qa_binary_encoding(examples, target, model_name):
    sentences, ce_labels, bce_labes = [], [], []
    questions = []
    tokenizer = BertTokenizer.from_pretrained(model_name)
    for example in tqdm(examples, desc='Processing data'):
        sentences.append(example.content)
        if example.label == target:
            ce_labels.append(1)
            bce_labes.append([1])
        else:
            ce_labels.append(0)
            bce_labes.append([0])
        questions.append(TRANS_DIC[target])
    print('===== Encoding... =====')
    encoded = tokenizer(questions, sentences, truncation=True, padding=True)
    input_ids = encoded['input_ids']
    attention_masks = encoded['attention_mask']
    token_type_ids = encoded['token_type_ids']
    feature = Feature(input_ids, attention_masks, token_type_ids, bce_labels=bce_labes, ce_labels=ce_labels)
    return feature


def joint_encoding(examples, model_name):
    sentences, bce_labels, ce_labels = [], [], []
    questions = []
    tokenizer = BertTokenizer.from_pretrained(model_name)
    for example in tqdm(examples, desc='Processing data'):
        for word, n_word in zip(TRANS_DIC.keys(), N_TRANS_DIC.keys()):
            if example.label == word:
                sentences.append(example.content)
                bce_labels.append([1])
                ce_labels.append(1)
                questions.append(TRANS_DIC[word])
            else:
                sentences.append(example.content)
                bce_labels.append([0])
                ce_labels.append(0)
                questions.append(TRANS_DIC[word])
            if example.label == n_word:
                sentences.append(example.content)
                bce_labels.append([0])
                ce_labels.append(0)
                questions.append(N_TRANS_DIC[word])
            else:
                sentences.append(example.content)
                bce_labels.append([1])
                ce_labels.append(1)
                questions.append(N_TRANS_DIC[word])
    print('===== Encoding... =====')
    encoded = tokenizer(questions, sentences, truncation=True, padding=True)
    input_ids = encoded['input_ids']
    attention_masks = encoded['attention_mask']
    token_type_ids = encoded['token_type_ids']
    feature = Feature(input_ids, attention_masks, token_type_ids, bce_labels=bce_labels, ce_labels=ce_labels)
    return feature


def joint_binary_encoding(examples, target, model_name):
    sentences, ce_labels = [], []
    questions = []
    tokenizer = BertTokenizer.from_pretrained(model_name)
    for example in tqdm(examples, desc='Processing data'):
        if example.label == target:
            ce_labels.append(1)
            sentences.append(example.content)
            questions.append(TRANS_DIC[target])
            ce_labels.append(0)
            sentences.append(example.content)
            questions.append(N_TRANS_DIC[target])
        else:
            ce_labels.append(0)
            sentences.append(example.content)
            questions.append(TRANS_DIC[target])
            ce_labels.append(1)
            sentences.append(example.content)
            questions.append(N_TRANS_DIC[target])
    encoded = tokenizer(questions, sentences, truncation=True, padding=True)
    input_ids = encoded['input_ids']
    attention_masks = encoded['attention_mask']
    token_type_ids = encoded['token_type_ids']
    feature = Feature(input_ids, attention_masks, token_type_ids, bce_labels=None, ce_labels=ce_labels)
    return feature


def zero_shot_encoding(examples, lack, model_name):
    sentences, bce_labels, ce_labels = [], [], []
    questions = []
    tokenizer = BertTokenizer.from_pretrained(model_name)
    if lack in TRANS_DIC.keys():
        del TRANS_DIC[lack]
    for example in tqdm(examples, desc='Processing data'):
        for word in TRANS_DIC.keys():
            if example.label != lack:
                sentences.append(example.content)
                questions.append(TRANS_DIC[word])
                if example.label == word:
                    bce_labels.append([1])
                    ce_labels.append(1)
                else:
                    bce_labels.append([0])
                    ce_labels.append(0)
    print('===== Encoding... =====')
    encoded = tokenizer(questions, sentences, truncation=True, padding=True)
    input_ids = encoded['input_ids']
    attention_masks = encoded['attention_mask']
    token_type_ids = encoded['token_type_ids']
    feature = Feature(input_ids, attention_masks, token_type_ids, bce_labels=bce_labels, ce_labels=ce_labels)
    return feature


def few_shot_encoding(examples, target, model_name, num):
    sentences, bce_labels, ce_labels = [], [], []
    _500_examples = []
    questions = []
    tokenizer = BertTokenizer.from_pretrained(model_name)
    for example in tqdm(examples, desc='Picking examples'):
        if example.label == target:
            _500_examples.append(example)
        if len(_500_examples) == num:
            break
    assert len(_500_examples) == num
    for example in tqdm(_500_examples, desc='Processing data'):
        for word in TRANS_DIC.keys():
            sentences.append(example.content)
            if word == target:
                bce_labels.append([1])
                ce_labels.append(1)
            else:
                bce_labels.append([0])
                ce_labels.append(0)
            questions.append(TRANS_DIC[word])
    assert len(sentences) == len(bce_labels) == len(ce_labels) == len(questions)
    print('===== Encoding... =====')
    encoded = tokenizer(questions, sentences, truncation=True, padding=True)
    input_ids = encoded['input_ids']
    attention_masks = encoded['attention_mask']
    token_type_ids = encoded['token_type_ids']
    feature = Feature(input_ids, attention_masks, token_type_ids, bce_labels=bce_labels, ce_labels=ce_labels)
    return feature


def pick_examples(examples, target, num):
    _num_examples = []
    for example in tqdm(examples, desc='Picking examples'):
        if example.label == target:
            _num_examples.append(example)
        if len(_num_examples) == num:
            break
    assert len(_num_examples) == num
    return _num_examples


def pick_train_examples(target, num):
    n_picked_examples = []
    p_picked_examples = []
    train_examples = example_reader('./data/train_contents.txt')
    for example in tqdm(train_examples, desc='Picking negative examples'):
        if example.label != target:
            n_picked_examples.append(example)
        if len(n_picked_examples) == num:
            break
    for example in tqdm(train_examples, desc='Picking positive examples'):
        if example.label == target:
            p_picked_examples.append(example)
        if len(p_picked_examples) == num:
            break
    assert len(p_picked_examples) == num
    return n_picked_examples + p_picked_examples
