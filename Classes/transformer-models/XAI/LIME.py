import torch
from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer

class_labels = ['0', '1', '2', '3', '4']
sample_text = "As a user I want to be able to provide the partitioning logic for a named destination so that i can control the ordering of outbound messages"

device = torch.device('cuda')
MODEL_NAME = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5, output_attentions = False, output_hidden_states = False)
model.to(device)

savedmodelname = "/content/drive/MyDrive/reserach/A_final_BERT_model.pth"
model.load_state_dict(torch.load(savedmodelname), strict=False)
model.eval()

tokens = tokenizer.encode_plus(
    sample_text,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
input_ids = tokens['input_ids'].cuda()
attention_mask = tokens['attention_mask'].cuda()

with torch.no_grad():
    outputs = model(input_ids, attention_mask)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

explainer = LimeTextExplainer(class_names=class_labels)

def predict_fn(texts):
    inputs = tokenizer.batch_encode_plus(
        texts,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to("cuda")
    attention_mask = inputs['attention_mask'].to("cuda")
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs.logits.cpu()
    probs = torch.softmax(logits, dim=1)  
    return probs.numpy()

explanation = explainer.explain_instance(
    sample_text,
    predict_fn,
    num_features=10,
    top_labels=1
)

print("The predicted class:", predicted_class)

output_html = 'lime.html'
explanation.save_to_file(output_html)