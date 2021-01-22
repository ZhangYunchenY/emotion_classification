import torch
import numpy as np
import torch.nn as nn
from sklearn import metrics
from emotion_classification.data_processer import *
from emotion_classification.model import model_bce as cls
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup


EPOCH = 4
BATCH_SIZE = 50
LOG_PATH = './log'
MODEL_NAME = 'bert-base-chinese'
TRAIN_PATH = './data/train_features.pkl'
DEV_PATH = './data/dev_features.pkl'
MODEL_SAVE_PATH = './model'

def train(train_dataloader, dev_dataloader):
    print('===== Loading model... =====')
    config = BertConfig.from_pretrained(MODEL_NAME)
    model = cls.BertForClassification(config, num_labels=7)
    model.cuda()
    # optimizer and scheduler
    total_step = EPOCH * len(train_dataloader)
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_step)
    # tensor board
    tensorboard_writer = SummaryWriter(LOG_PATH)
    # var
    dev_loss_epoch_count = 0
    # training
    for i in range(0, EPOCH):
        epoch_loss = 0
        for step, batch in tqdm(enumerate(train_dataloader), desc='Training', total=len(train_dataloader)):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch = tuple(t.cuda() for t in batch)
            b_input_ids, b_attention_masks, b_token_type_ids, b_labels = batch
            output = model(b_input_ids, labels=b_labels,
                           token_type_ids=b_token_type_ids,
                           attention_mask=b_attention_masks)
            loss = output
            epoch_loss += loss.item()
            tensorboard_writer.add_scalar('train_loss', epoch_loss/(step+1), step+i*len(train_dataloader))
            tensorboard_writer.flush()
            loss.backward()
            # update parameters
            optimizer.step()
            scheduler.step()

        # validation
        model.eval()
        dev_epoch_loss = 0
        predictions = []
        for batch in tqdm(dev_dataloader, desc='Validation', total=len(dev_dataloader)):
            batch = tuple(t.cuda() for t in batch)
            d_input_ids, d_attention_masks, d_token_type_ids, d_labels = batch
            with torch.no_grad():
                d_output = model(d_input_ids, labels=None,
                                 token_type_ids=d_token_type_ids,
                                 attention_mask=d_attention_masks)
                logits = d_output
                # calculate loss
                d_labels = d_labels.float()
                loss_fct = nn.BCELoss()
                bce_loss = loss_fct(logits, d_labels)
                dev_epoch_loss += bce_loss.item()
                # calculate predictions
                logits = logits.detach().cpu().numpy()
                for logit in logits:
                    index = np.argmax(logit)
                    predictions.append(index)
        dev_epoch_loss /= len(dev_dataloader)
        precision = metrics.precision_score(dev_features.ce_labels, predictions, average='macro')
        recall = metrics.recall_score(dev_features.ce_labels, predictions, average='macro')
        f1 = metrics.f1_score(dev_features.ce_labels, predictions, average='macro')
        tensorboard_writer.add_scalar('dev_epoch_loss', dev_epoch_loss, dev_loss_epoch_count)
        tensorboard_writer.add_scalar('dev_precision', precision, dev_loss_epoch_count)
        tensorboard_writer.add_scalar('dev_recall', recall, dev_loss_epoch_count)
        tensorboard_writer.add_scalar('dev_f1', f1, dev_loss_epoch_count)
        tensorboard_writer.flush()
        dev_loss_epoch_count += 1
    # complete
    torch.save(model.state_dict(), MODEL_SAVE_PATH)


if __name__ == '__main__':
    train_features = feature_reader(TRAIN_PATH)
    dev_features = feature_reader(DEV_PATH)
    train_dataloader = creat_dataloader(BATCH_SIZE, train_features.input_ids,
                                        train_features.attention_masks,
                                        train_features.token_type_ids,
                                        train_features.bce_labels)
    dev_dataloader = creat_dataloader(BATCH_SIZE, dev_features.input_ids,
                                        dev_features.attention_masks,
                                        dev_features.token_type_ids,
                                        dev_features.bce_labels)
    train(train_dataloader, dev_dataloader)