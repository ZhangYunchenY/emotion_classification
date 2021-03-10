import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys

sys.path.append('..')
from emotion_classification.data_processer import *
from emotion_classification.model import model_bce as cls
from sklearn import metrics
from transformers import BertConfig

BATCH_SIZE = 1050
TARGET = 'fear'
MODEL_NAME = 'bert-base-chinese'
TRAIN_PATH = './data/train_contents.txt'
DEV_PATH = './data/dev_contents.txt'
MODEL_PATH = '../model/motion_few_shot_disgust_100.pt'
DEV_FEATURES_PATH = './data/dev_features.pkl'


def validation(dataloader):
    print('===== Loading model =====')
    state_dict = torch.load(MODEL_PATH)
    config = BertConfig.from_pretrained('bert-base-chinese')
    model = cls.BertForClassification(config, num_labels=1)
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    predictions = []
    for batch in tqdm(dataloader, desc='Validation', total=len(dataloader)):
        batch = tuple(t.cuda() for t in batch)
        d_input_ids, d_attention_masks, d_token_type_ids, d_labels = batch
        with torch.no_grad():
            d_output = model(d_input_ids, labels=None,
                             token_type_ids=d_token_type_ids,
                             attention_mask=d_attention_masks)
            logits = d_output
            logits = logits.detach().cpu().numpy()

            # for binary
            for logit in logits:
                if logit[0] > 0.5:
                    predictions.append(1)
                else:
                    predictions.append(0)

            # # for joint
            # probs = []
            # for logit in logits:
            #     probs.append(logit[0])
            # y_probs = np.array(probs[::2])
            # n_probs = np.array(probs[1::2])
            # d_probs = y_probs - n_probs

            # # joint norm
            # i = 0
            # while i < len(d_probs) // 7:
            #     d_prob = d_probs[7*i:7*(i+1)]
            #     index = np.argmax(d_prob)
            #     predictions.append(index)
            #     i += 1

            # # joint rouge
            # for prob in d_probs:
            #     if prob > 0.:
            #         predictions.append(1)
            #     else:
            #         predictions.append(0)

            # # norm validation
            # i = 0
            # probs = []
            # for logit in logits:
            #     probs.append(logit[0])
            # while i < len(probs) // 7:
            #     prob = probs[7*i:7*(i+1)]
            #     index = np.argmin(prob)
            #     predictions.append(index)
            #     i += 1

            # # for multi labels
            # for logit in logits:
            #     index = np.argmax(logit)
            #     predictions.append(index)

    precision = metrics.precision_score(dev_features.ce_labels, predictions, average='binary')
    recall = metrics.recall_score(dev_features.ce_labels, predictions, average='binary')
    f1 = metrics.f1_score(dev_features.ce_labels, predictions, average='binary')
    print(f'Precision={precision}')
    print(f'Recall={recall}')
    print(f'F1={f1}')
    print(metrics.confusion_matrix(dev_features.ce_labels, predictions))
    TP, TN, FP, FN = 0, 0, 0, 0
    i = 0
    while i < len(predictions):
        if predictions[i] == 1 and dev_features.ce_labels[i] == 1:
            TP += 1
        elif predictions[i] == 1 and dev_features.ce_labels[i] == 0:
            FP += 1
        elif predictions[i] == 0 and dev_features.ce_labels[i] == 1:
            FN += 1
        else:
            TN += 1
        i += 1
    print(f'TP:{TP}')
    print(f'FP:{FP}')
    print(f'FN:{FN}')
    print(f'TN:{TN}')
    print(f'ACC:{(TP + TN) / (TP + TN + FP + FN)}')


if __name__ == '__main__':
    # dev_examples = example_reader(DEV_PATH)

    dev_examples = adjust_dataset(DEV_PATH, TARGET)
    dev_features = qa_binary_encoding(dev_examples, TARGET, MODEL_NAME)

    # dev_features = feature_reader('./data/dev_features_lack_of_happiness.pkl')

    # dev_features = qa_encoding(dev_examples, MODEL_NAME)
    # dev_features_norm = feature_reader(DEV_FEATURES_PATH)
    # dev_features_norm = qa_encoding(dev_examples, MODEL_NAME)
    dev_dataloader = creat_dataloader(BATCH_SIZE, dev_features.input_ids,
                                      dev_features.attention_masks,
                                      dev_features.token_type_ids,
                                      dev_features.ce_labels)
    validation(dev_dataloader)
