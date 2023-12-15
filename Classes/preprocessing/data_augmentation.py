import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import nlpaug.augmenter.word as nlpaw
import nltk

nltk.download('stopwords')

def remove_punctuations(data):
    punct_tag=re.compile(r'[^\w\s]')
    data=punct_tag.sub(r' ',data)
    return data

storypoint_mapping = {
    1: 0,
    2: 1,
    3: 2,
    5: 3,
    8: 4
}

df = pd.read_csv("./filtered_dataset2.csv")
df = df.dropna()
df = df.dropna()
df = df[df["storypoint"] != 13]
df['storypoint'] = df['storypoint'].map(storypoint_mapping)

trainData, testData = train_test_split(df, test_size=0.3, random_state=0)
testData, validationData = train_test_split(testData, test_size=0.5, random_state=0)  # 0.25 x 0.8 = 0.2

print("Number of unique labels: \n", trainData['storypoint'].value_counts())
print("Number of unique labels: \n", testData['storypoint'].value_counts())
print("Number of unique labels: \n", validationData['storypoint'].value_counts())

testData.to_csv('testdata.csv', index=False)
validationData.to_csv('Tvaldata.csv', index=False)
trainData.to_csv('traindata.csv', index=False)


trainData['text'] = trainData['title'] + ' ' + trainData['description']
trainData.drop(['title', 'description'], axis=1, inplace=True)

X = trainData['text']
y = trainData['storypoint']

max_samples = 7000

augmented_data = pd.DataFrame(columns=['text', 'storypoint'])

def augment_sentence(sentence, aug, num_threads):
    return aug.augment(sentence, num_thread=num_threads)
    
def augment_text(df, aug, num_threads, num_times, spval):
    # Get rows of data to augment
    to_augment = df[df['storypoint']==spval]
    to_augmentX = to_augment['text']
    to_augmentY = np.full(len(to_augmentX.index) * num_times, spval, dtype=np.int8)
    
    # Build up dictionary containing augmented data
    aug_dict = {'text':[], 'storypoint':to_augmentY}
    for i in tqdm(range(num_times)):
        augX = [augment_sentence(x, aug, num_threads) for x in to_augmentX]
        aug_dict['text'].extend(augX)
    
    # Build DataFrame containing augmented data
    aug_df = pd.DataFrame.from_dict(aug_dict)

    namesfornum = str(spval) + "augmented_dataset.csv"
    aug_df.to_csv(namesfornum, index=False)
    
    return df.append(aug_df, ignore_index=True).sample(frac=1, random_state=42)
    

aug10p = nlpaw.ContextualWordEmbsAug(model_path='bert-base-uncased', aug_min=1, aug_p=0.2, action="substitute")

spvals = [1,2,3,5,8]
numtimes = [2,2,3,3,5]
for i in range(5):
    trainData = augment_text(trainData, aug10p, num_threads=20, num_times=numtimes[i], spval=spvals[i])

trainData.to_csv('Aug_Data_for_training.csv', index=False)