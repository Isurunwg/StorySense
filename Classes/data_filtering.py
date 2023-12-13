import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

dataset = pd.read_csv('.dataset.csv')
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
dataset = dataset.dropna().reset_index(drop=True)
dataset['storypoint'] = dataset['storypoint'].map(storypoint_mapping)

# Load pre-trained SBERT model
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Combine title and description into a single text column
dataset['text'] = dataset['title'] + ' ' + dataset['description']

# Create a copy of the original dataframe to store the filtered data
filtered_df = dataset.copy()

# Encode text data using SBERT
text_embeddings = sbert_model.encode(dataset['text'].tolist())

# Threshold for similarity (adjust as needed)
threshold = 0.9

deleteIndex = []

# Iterate through each data item
for i in range(len(dataset)):
    # Initialize a list to store indices of similar items
    similar_indices = []

    # Find similar titles or descriptions with different storypoint values
    for j in range(i + 1, len(dataset)):
        if cosine_similarity([text_embeddings[i]], [text_embeddings[j]]) > threshold and dataset.iloc[i]['storypoint'] != dataset.iloc[j]['storypoint']:
            similar_indices.append(j)

    # Remove all similar items except for the one with the highest storypoint value
    if similar_indices:
        max_storypoint_index = max(similar_indices, key=lambda x: dataset.iloc[x]['storypoint'])
        similar_indices.remove(max_storypoint_index)
        deleteIndex.extend(similar_indices)

unique_list = list(set(deleteIndex))
filtered_df = filtered_df.drop(deleteIndex, axis=0).reset_index(drop=True)
filtered_df.to_csv('filtered_dataset.csv', index=False)

print(filtered_df)
