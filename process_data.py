from emotion_classification.data_processer import *


TARGET = 'qa_w'
MODEL_NAME = 'bert-base-chinese'
DATA_PATH = './data/OCEMOTION_train1128.csv'
SAVA_PATH = './data/'
TRAIN_PATH = './data/train_contents.txt'
DEV_PATH = './data/dev_contents.txt'
TRAIN_SAVE_PATH = './data/train_features_' + TARGET + '.pkl'
DEV_SAVE_PATH = './data/dev_features_' + TARGET + '.pkl'


if __name__ == '__main__':
    split_dataset(DATA_PATH, SAVA_PATH)
    train_examples = example_reader(TRAIN_PATH)
    dev_examples = example_reader(DEV_PATH)
    data_analysis(train_examples)
    data_analysis(dev_examples)
    train_features = qa_encoding(train_examples, MODEL_NAME)
    dev_features = qa_encoding(dev_examples, MODEL_NAME)
    feature_writer(TRAIN_SAVE_PATH, train_features)
    feature_writer(DEV_SAVE_PATH, dev_features)
