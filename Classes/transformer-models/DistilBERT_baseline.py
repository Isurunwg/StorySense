import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset,  RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

def preprocess_dataset(data_file):
    data = pd.read_csv(data_file)
    data = data.drop(columns=['issuekey'])
    data['storypoint'] = data['storypoint'].astype(int)
    data = data.dropna()
    return data

def dataset_columns(dataset):
      return len(dataset.columns)

def prepare_data_for_model(data, number_of_cols, max_length):
    if number_of_cols == 3:
        text = (data['title'] + " " + data['description']).tolist()
    else:
        text = (data['title'] + " " + data['description'] + " " + data['priority'] + " " + data['type']).tolist()

    # Tokenize the text
    encoded_data = tokenize_data(tokenizer, text, max_length)

    # Extract input_ids, attention_mask, and labels
    input_ids = encoded_data['input_ids'].cuda()
    attention_mask = encoded_data['attention_mask'].cuda()
    labels = torch.tensor(data['storypoint'].tolist()).cuda()

    # Create DataLoader
    dataset = TensorDataset(input_ids, attention_mask, labels)
    return dataset

# Function to tokenize the input data
def tokenize_data(tokenizer, text_list, max_seq_length):
    tokenized_data = tokenizer.batch_encode_plus(
        text_list,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors='pt',
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,

    )
    return tokenized_data

# Function to train the BERT model
def train_model(model, train_dataloader, optimizer, classifier_optimizer, val_dataloader, class_weights_train, class_weights_eval, scheduler, num_epochs=4):
    model.to(device)
    for epoch in range(num_epochs):
        print("epoch >>", epoch+1)

        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            input_ids, attention_mask, labels = batch

            optimizer.zero_grad()
            model.zero_grad()
            classifier_optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)

            # criterion = nn.CrossEntropyLoss(weight=class_weights_train)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([class_weights_train[0],class_weights_train[1], class_weights_train[2], class_weights_train[3], class_weights_train[4]], dtype=torch.float32).cuda())
            
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            classifier_optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}')
        # evaluate_model(model, train_dataloader, "train >>", )
        evaluate_model(model, val_dataloader, "eval >>", class_weights_eval)
        print(" ")



def evaluate_model(model, dataloader, set_name, class_weights_eval):
    model.to(device)
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        absolute_errors = []

        for batch in dataloader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([class_weights_eval[0],class_weights_eval[1], class_weights_eval[2], class_weights_eval[3], class_weights_eval[4]], dtype=torch.float32).cuda())


            loss = criterion(outputs.logits, labels)

            test_loss += loss.item()

            _, predicted_labels = torch.max(outputs.logits, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted_labels == labels).sum().item()

            absolute_errors.extend(torch.abs(predicted_labels - labels).cpu().numpy())
            # mae += torch.abs(predicted_labels - labels).sum().item()

        accuracy = 100 * correct_predictions / total_predictions
        # avg_loss = test_loss / len(dataloader)
        avg_loss = test_loss
        avg_mae = np.mean(absolute_errors)
        # avg_mae = mae / total_predictions
        md_ae = np.median(absolute_errors)

        print(f'{set_name} Loss: {avg_loss:.4f} | {set_name} Accuracy: {accuracy:.2f}% | MAE: {avg_mae:.4f} | MdAE: {md_ae:.4f}')

def testmodel(model, dataloader, set_name):
    correct_per_class = {}
    total_per_class = {}
    model.to(device)
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_predicted_labels = []
    all_true_labels = []
    absolute_errors = []

    # Iterate through the DataLoader
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = outputs.loss
            # loss = nn.CrossEntropyLoss(outputs.logits, labels)
            total_predictions += labels.size(0)
            test_loss += loss.item()
            probabilities = F.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            # predicted_labels = torch.argmax(probabilities, dim=1)
            correct_predictions += (predicted_class == labels).sum().item()
            absolute_errors.extend(torch.abs(predicted_class - labels).cpu().numpy())

            accuracy = 100 * correct_predictions / total_predictions
            avg_mae = np.mean(absolute_errors)
            md_ae = np.median(absolute_errors)

            incorrect_predictions = total_predictions - correct_predictions
            print("Test evaluation for model", i+1)
            print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {accuracy:.2f}% | MAE: {avg_mae:.4f} | MdAE: {md_ae}')
    
            all_predicted_labels.extend(predicted_class.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            with open('predicted_labels.txt', 'w') as f:
                for label in all_predicted_labels:
                    f.write(f'{label}\n')

            with open('true_labels.txt', 'w') as f:
                for label in all_true_labels:
                    f.write(f'{label}\n')
                        
            correct_predictions += (predicted_class == labels).sum().item()
            total_predictions += labels.size(0)

            for true_label, predicted_label in zip(labels, predicted_class):
                true_label = true_label.item()
                predicted_label = predicted_label.item()

                # Update dictionaries
                if true_label not in correct_per_class:
                    correct_per_class[true_label] = 0
                if true_label not in total_per_class:
                    total_per_class[true_label] = 0

                total_per_class[true_label] += 1
                if true_label == predicted_label:
                    correct_per_class[true_label] += 1

    # Calculate accuracy for each class
    class_accuracies = {}
    for class_label, correct_count in correct_per_class.items():
        total_count = total_per_class[class_label]
        accuracy = correct_count / total_count if total_count > 0 else 0
        class_accuracies[class_label] = accuracy

    # Print or use the class accuracies as needed
    for class_label, accuracy in class_accuracies.items():
        print(f"Class {class_label}: Accuracy = {accuracy:.2%}")



global device

DATA_FILE_PATH  = './Datasets/dataset_A.csv'
MODEL_NAME = 'bert-base-uncased'
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16
STORY_POINTS = [1, 2, 3, 5, 8]

LABEL_MAPPING = {
    1: 0,
    2: 1,
    3: 2,
    5: 3,
    8: 4,
}

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = preprocess_dataset(DATA_FILE_PATH)
df = df[df['storypoint'].isin([1,2,3,5,8])]

train_dataset, tdf = train_test_split(df, test_size=0.3, random_state=42)
val_dataset, test_dataset = train_test_split(tdf, test_size=0.5, random_state=42)

test_dataset['storypoint'] = test_dataset['storypoint'].map(LABEL_MAPPING)
val_dataset['storypoint'] = val_dataset['storypoint'].map(LABEL_MAPPING)
train_dataset['storypoint'] = train_dataset['storypoint'].map(LABEL_MAPPING)

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5, output_attentions = False, output_hidden_states = False)

number_of_cols = dataset_columns(df)

class_weights_train = {}
class_weights_eval = {}
class_weights_test = {}

class_frequencies = np.unique(train_dataset['storypoint'], return_counts=True)[1]
print("class_frequencies:", class_frequencies)
for i in range(len(class_frequencies)):
    class_weights_train[i] = (class_frequencies[0] + class_frequencies[1] + class_frequencies[2] + class_frequencies[3] + class_frequencies[4]) / class_frequencies[i]

class_frequencies = np.unique(val_dataset['storypoint'], return_counts=True)[1]
for i in range(len(class_frequencies)):
    class_weights_eval[i]  = (class_frequencies[0] + class_frequencies[1] + class_frequencies[2] + class_frequencies[3] + class_frequencies[4])/ class_frequencies[i]

class_frequencies = np.unique(test_dataset['storypoint'], return_counts=True)[1]
for i in range(len(class_frequencies)):
    class_weights_test[i]  = (class_frequencies[0] + class_frequencies[1] + class_frequencies[2] + class_frequencies[3] + class_frequencies[4])/ class_frequencies[i]


tdataset = prepare_data_for_model(train_dataset, number_of_cols, MAX_SEQ_LENGTH)
vdataset = prepare_data_for_model(val_dataset, number_of_cols, MAX_SEQ_LENGTH)
testdataset = prepare_data_for_model(test_dataset, number_of_cols, MAX_SEQ_LENGTH)

# Define batch size and create DataLoader instances
train_dataloader = DataLoader(tdataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(vdataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

# optimizer = AdamW(model.parameters(), lr=2e-5)
model.dropout.p = 0.3
weight_decay = 0.001

optimizer = AdamW(model.parameters(),
                lr = 2e-5, 
                eps = 2e-5,
                weight_decay=weight_decay
                )
classifier_optimizer = AdamW(model.classifier.parameters(), lr=2e-5)

epochs = 4
total_steps = len(train_dataloader) * epochs
warmup_steps = int(0.1 * len(train_dataloader))
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = total_steps)

print("class_weights_train: ", class_weights_train)
print("class_weights_eval: ", class_weights_eval)
print("class_weights_test: ", class_weights_test)

print("train started")

train_model(model, train_dataloader, optimizer, classifier_optimizer, val_dataloader, class_weights_train, class_weights_eval, scheduler)

model.save_pretrained('saved_model')
tokenizer.save_pretrained('saved_model')
savedmodelname = "BERT_model.pth"
torch.save(model.state_dict(), savedmodelname)

val_loss = evaluate_model(model, test_dataloader, "test", class_weights_test)
model.load_state_dict(torch.load(savedmodelname))
test_loss = testmodel(model, test_dataloader, "Test")

model.save_pretrained('saved_model')
tokenizer.save_pretrained('saved_model')
torch.save(model.state_dict(), savedmodelname)
print("Trained model saved successfully!")