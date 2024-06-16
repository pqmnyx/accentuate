import spacy
from spacy import load
from spacy.lang.ru.examples import sentences
from spacy.lang.ru import Russian

from nltk.corpus import stopwords
import nltk



nltk.download('stopwords')
stopwords_ru = stopwords.words("russian")

nlp = Russian()
load_model = load("ru_core_news_sm")

