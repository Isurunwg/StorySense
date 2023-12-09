import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, median_absolute_error, accuracy_score, confusion_matrix
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

storypoint_mapping = {
    1: 0,
    2: 1,
    3: 2,
    5: 3,
    8: 4
}

dataset = pd.read_csv("./data.csv")

dataset['title'] = dataset['title'].astype(str)
dataset['description'] = dataset['description'].astype(str)
dataset['storypoint'] = pd.to_numeric(dataset['storypoint'], errors='coerce')

storypoint_mapping = {
    1: 0,
    2: 1,
    3: 2,
    5: 3,
    8: 4
}

dataset = dataset[dataset["storypoint"] != 13]
dataset['storypoint'] = dataset['storypoint'].map(storypoint_mapping)

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Preprocess the text data
def preprocess_text(text):
    return simple_preprocess(text, deacc=True)

# Train word2vec model on the training data
sentences = [preprocess_text(text) for text in train_data['title'] + ' ' + train_data['description']]
word2vec_model = Word2Vec(sentences, min_count=1)

# Function to convert text to word2vec embeddings
def text_to_word2vec(text):
    words = preprocess_text(text)
    embeddings = []
    for word in words:
        if word in word2vec_model.wv:
            embeddings.append(word2vec_model.wv[word])
    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

# Convert the training data to word2vec embeddings
train_embeddings = np.array([text_to_word2vec(text) for text in train_data['title'] + ' ' + train_data['description']])
train_labels = train_data['storypoint']

# Convert the testing data to word2vec embeddings
test_embeddings = np.array([text_to_word2vec(text) for text in test_data['title'] + ' ' + test_data['description']])
test_labels = test_data['storypoint']

# Train the SVM model
svm_model = SVC(kernel='rbf', C=5000, gamma=1, class_weight='balanced')
svm_model.fit(train_embeddings, train_labels)

# Evaluate the model
test_predictions = svm_model.predict(test_embeddings)
test_mae = mean_absolute_error(test_labels, test_predictions)
test_mdae = median_absolute_error(test_labels, test_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)
classwise_accuracy = accuracy_score(test_labels, test_predictions, normalize=False)
confusion_mat = confusion_matrix(test_labels, test_predictions)

# Calculate the test loss (hinge loss)
test_loss = np.mean(np.maximum(0, 1 - np.multiply(test_labels, test_predictions)))

# Print the evaluation results
print('Test MAE:', test_mae)
print('Test MdAE:', test_mdae)
print('Test Overall Accuracy:', test_accuracy)
print('Test Class-wise Accuracy:', classwise_accuracy)
print('Test Loss:', test_loss)
print('Confusion Matrix:')
print(confusion_mat)