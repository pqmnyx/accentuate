import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from spacy import load
from spacy.lang.ru.examples import sentences
from spacy.lang.ru import Russian

from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, NewsNERTagger,NamesExtractor, Doc

input_data = input()
author_data = pd.DataFrame({'text': [input_data]})
author_data.to_csv(r'text.csv', index=False)
print(author_data)

# лемматизация
nlp = Russian()
load_model = load("ru_core_news_sm")
lemma = []

for doc in load_model.pipe(author_data["text"].values):
    lemma.append([n.lemma_ for n in doc])

author_data['text_clean_lemma'] = lemma
author_data[['text', 'text_clean_lemma']].head()
author_data.to_csv(r'text.csv', index=False)
#Удаление стоп-слов и преобразование в строку
stopwords_ru = stopwords.words("russian")


author_data['text_clean_lemma'] = author_data['text_clean_lemma'].apply(lambda x: [item for item in x if item not in stopwords_ru])
author_data['text_clean_lemma_as_str'] = [' '.join(map(str, l)) for l in author_data['text_clean_lemma']]
print(author_data)

# Токенизация и морфологический анализ
author_data['text_clean_lemma_as_str'].dropna(inplace=True)
author_data_tokenized = word_tokenize(author_data['text_clean_lemma_as_str'])

print(author_data['text_clean_lemma'])
print(author_data_tokenized)