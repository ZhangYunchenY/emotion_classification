import torch
import torch.nn as nn
from transformers import BertModel
from emotion_classification.model.Focal_loss import FocalLoss


class BertForClassification(nn.Module):
    def __init__(self, config, num_labels=1):
        super(BertForClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext-large', config=config)
        # self.bert = BertModel.from_pretrained('bert-base-chinese', config=config, output_attentions=True)
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=None,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            labels = labels.float()
            # pos_weight = neg_count / pos_count (type=float32)
            weight = torch.tensor(2.)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=weight)
            loss = loss_fct(logits, labels)
            return loss
        else:
            sigmoid_fct = nn.Sigmoid()
            logits = sigmoid_fct(logits)
            return logits
