import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import shap
import shap.plots
import scipy as sp

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

def f(x):
    tv = torch.tensor(
        [
            tokenizer.encode(v, padding="max_length", max_length=128, truncation=True)
            for v in x
        ]
    ).cuda()
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:, 1]) 
    return val


explainer = shap.Explainer(f, tokenizer)

# shap_values = explainer(sample_text, fixed_context=1, batch_size=2)
shap_values = explainer([sample_text])
shap.plots.text(shap_values)
shap.plots.bar(shap_values.abs.mean(0))

summary_plot = shap.plots.text(shap_values, display=False)

with open('shap.html', 'w') as f:
    f.write(summary_plot)