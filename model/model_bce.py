import torch.nn as nn
from transformers import BertModel


class BertForClassification(nn.Module):
    def __init__(self, config, num_labels=1):
        super(BertForClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-chinese', config=config)
        # self.bert = self.bert.from_pretrained(config.bert_model, )

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
            # 如果将label传入，将返回损失，使用BCEWithLogitsLoss
            # BCELogitsLoss中，首先sigmoid，然后和label对比计算损失
            # 将label向量转换为float
            labels = labels.float()
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            return loss
        else:
            # 经过softMax和sigmoid之后的向量
            sigmoid_fct = nn.Sigmoid()
            logits = sigmoid_fct(logits)
            return logits
