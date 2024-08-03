import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.parsing.preprocessing import STOPWORDS
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import nltk

nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)


def calculate_mtld(text, threshold=0.72):
    words = word_tokenize(text.lower())
    ttr = 1.0
    word_count = 0
    types = set()
    factors = 0

    for word in words:
        word_count += 1
        types.add(word)
        ttr = len(types) / word_count
        if ttr <= threshold:
            factors += 1
            word_count = 0
            types = set()
            ttr = 1.0

    if word_count > 0:
        factors += (1 - ttr) / (1 - threshold)

    return len(words) / factors if factors > 0 else 0


def analyze_document_similarity(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


def analyze_topics(documents, num_topics=5):
    texts = [[word for word in document.lower().split() if word not in STOPWORDS]
             for document in documents]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaMulticore(
        corpus=corpus, id2word=dictionary, num_topics=num_topics)
    topics = lda_model.print_topics()
    # Retourne seulement la chaîne de mots pour chaque topic
    return [topic[1] for topic in topics]


def analyze_lexical_richness(text):
    words = word_tokenize(text.lower())
    unique_words = set(words)
    word_count = len(words)
    unique_word_count = len(unique_words)

    mtld = calculate_mtld(text)
    guiraud = unique_word_count / np.sqrt(word_count) if word_count > 0 else 0

    return {
        "MTLD": mtld,
        "Guiraud Index": guiraud
    }


def extract_key_phrases(text, top_n=5):
    words = word_tokenize(text.lower())
    tagged = pos_tag(words)
    noun_phrases = []
    current_phrase = []

    for word, tag in tagged:
        if tag.startswith('N'):
            current_phrase.append(word)
        elif current_phrase:
            noun_phrases.append(' '.join(current_phrase))
            current_phrase = []

    if current_phrase:
        noun_phrases.append(' '.join(current_phrase))

    phrase_scores = {phrase: text.count(phrase)
                     for phrase in set(noun_phrases)}
    return sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


def perform_advanced_analysis(file_stats, all_text):
    # Utilise le texte complet, pas la longueur
    documents = [stat[1] for stat in file_stats]

    similarity_matrix = analyze_document_similarity(documents)
    topics = analyze_topics(documents)
    lexical_richness = analyze_lexical_richness(all_text)
    key_phrases = extract_key_phrases(all_text)

    return {
        "similarity_matrix": similarity_matrix,
        "topics": topics,
        "lexical_richness": lexical_richness,
        "key_phrases": key_phrases
    }
