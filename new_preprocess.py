# IMPORTING LIBRARIES:
import re
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

# Loading data:
# (opou 10 na valw 50000)
data = pd.read_csv('IMDB Dataset.csv')
#data = data1.iloc[:10, :10]
y = data.iloc[:, 1].values

# defining label encoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y)
y = y.reshape(50000, 1)


# defining tokenization object
tokenizer = nltk.ToktokTokenizer()

# defining list of stopwords
stop_list = nltk.corpus.stopwords.words('english')

# ---------------------------------------------------------------------------------------------------------------------
# TEXT DENOISING

# defining function for html strips


def html_parser(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

# defining function for square brackets


def remove_squarebr(text):
    return re.sub('\[[^]]*\]', '', text)


# defining function for noisy text


def denoise_text(text):
    text = html_parser(text)
    text = remove_squarebr(text)
    text = text.lower()
    return text

# applying the function:


data['review'] = data['review'].apply(denoise_text)

# ---------------------------------------------------------------------------------------------------------------------

# defining function for removing special characters


def remove_sc(text):
    char = r'[^a-zA-z0-9\s]'
    text = re.sub(char, '', text)
    return text

# applying the function:


data['review'] = data['review'].apply(remove_sc)

# definition of stemming, but it wasn't use because of its lack of explainability
'''
# defining function for text stemming


def stemming(text):
    ps = nltk.stem.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

# applying the function:


data['review'] = data['review'].apply(stemming)

'''
# defining function for removing stopwords:

sp = set(stopwords.words('english'))


def remove_stopwords(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stop_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# applying the function:


data['review'] = data['review'].apply(remove_stopwords)


# defining function for tokinaziton:


def token(text):
    text = nltk.word_tokenize(text)
    return text


# applying thn function:
data['review'] = data['review'].apply(token)

print(data['review'])



# building vocabulary:


def build_vocabulary(text):
    vocabulary = dict()

    fdist = nltk.FreqDist()

    for sentence in text:
        for word in sentence:
            fdist[word] += 1

    common_words = fdist.most_common(n=2000)

    for idx, word in enumerate(common_words):
        vocabulary[word[0]] = (idx+1)

    return vocabulary


Vocabulary = build_vocabulary(data['review'])

print(Vocabulary)


def word_ton_index(text):

    x_tokenized = list()

    for sentence in text:
        temp_sentence = list()
        for word in sentence:
            if word in Vocabulary.keys():
                temp_sentence.append(Vocabulary[word])
        x_tokenized.append(temp_sentence)
    return x_tokenized


indexes = word_ton_index(data['review'])

print(indexes)

row_length = []
for row in indexes:
    row_length.append(len(row))

max_length = max(row_length)
print(max_length)


def pad_sentences(text):

    pad_index = 0
    x_padded = list()

    for sentence in indexes:
        while len(sentence) < max_length:
            sentence.insert(len(sentence), pad_index)
        x_padded.append(sentence)

    x_padded = np.array(x_padded)

    return x_padded

x_padded = pad_sentences(data['review'])

def save_to_csv(x):
    df = np.concatenate((x, y), axis=1)
    df = pd.DataFrame(df)
    df = df.to_csv('preprocessednew.csv')
    return df

save_to_csv(x_padded)

