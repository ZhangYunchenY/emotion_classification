import torch.nn as nn
from transformers import BertModel


class BertForClassification(nn.Module):
    def __init__(self, config, num_labels=1):
        super(BertForClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.hidden_size = config.hidden_size
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        # self.dropout1 = nn.Dropout(0.4)
        self.classifier1 = nn.Linear(self.hidden_size, 4096)
        self.dropout2 = nn.Dropout(0.4)
        self.classifier2 = nn.Linear(4096, self.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=None,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout1(pooled_output)
        logits = self.classifier1(pooled_output)
        logits = self.dropout2(logits)
        logits = self.classifier2(logits)

        if labels is not None:
            # 如果将label传入，将返回损失，使用CELoss
            # 将label向量转换为long
            labels = labels.long()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
        else:
            # 经过softMax和softmax之后的向量
            softmax_fct = nn.Softmax(dim=1)
            logits = softmax_fct(logits)
            return logits
