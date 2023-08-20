from nltk.corpus import stopwords
from nltk import word_tokenize
import re
import sys


def clear_devamini_oku(text):
    temptext = text.split(".")
    if "Devamını" in temptext[-1]:
        text = temptext[:-1]
    return "".join(text)


def prepare_data(text):
    stopword_set = set(stopwords.words('turkish'))  # NLTK
    text = re.sub('[^a-zA-ZğĞüÜşŞıİöÖçÇ]', " ", text)
    text = text.lower()
    text = word_tokenize(text, language='turkish')
    text = [word for word in text if not word in stopword_set]
    return " ".join(text)
