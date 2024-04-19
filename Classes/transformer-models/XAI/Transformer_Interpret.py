import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from transformers_interpret import SequenceClassificationExplainer

class_labels = ['0', '1', '2', '3', '4']
sample_text = "As a user I want to be able to provide the partitioning logic for a named destination so that i can control the ordering of outbound messages"
savedmodelname = "/BERT_model.pth"

device = torch.device('cuda')
MODEL_NAME = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5, output_attentions = False, output_hidden_states = False)
model.to(device)

model.load_state_dict(torch.load(savedmodelname), strict=False)
model.eval()

cls_explainer = SequenceClassificationExplainer(
    model,
    tokenizer)

word_attributions = cls_explainer(sample_text)
print(word_attributions)
print(cls_explainer.predicted_class_name)
cls_explainer.visualize("transformer-interpret.html")