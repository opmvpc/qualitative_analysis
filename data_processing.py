import os
from collections import Counter
import docx
from nltk.tokenize import word_tokenize, sent_tokenize
from custom_stopwords import CUSTOM_STOPWORDS
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(nltk.corpus.stopwords.words(
        'french')).union(CUSTOM_STOPWORDS)
    return [word for word in tokens if word.isalpha() and len(word) > 2 and word not in stop_words]


def filter_questions(text):
    # Split text into lines and filter out lines starting with 'T:'
    lines = text.split('\n')
    filtered_lines = [
        line for line in lines if not line.strip().startswith('T:')]
    return " ".join(filtered_lines)


def read_and_preprocess_files(directory):
    all_words = []
    file_stats = []
    all_text = ""
    for filename in os.listdir(directory):
        if filename.endswith('.docx'):
            file_path = os.path.join(directory, filename)
            text = read_docx(file_path)
            text = filter_questions(text)  # Filter out the questions
            all_text += text + " "
            words = preprocess_text(text)
            all_words.extend(words)
            sentences = sent_tokenize(text)
            if len(sentences) > 0:
                avg_sentence_length = sum(len(sent.split())
                                          for sent in sentences) / len(sentences)
            else:
                avg_sentence_length = 0
            file_stats.append((filename, text, len(words),
                              len(sentences), avg_sentence_length))

    word_freq = Counter(all_words)
    return word_freq, file_stats, all_text
