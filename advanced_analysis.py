from similarity_analysis import compute_all_similarities
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.parsing.preprocessing import STOPWORDS
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import nltk
import networkx as nx
from nltk.tokenize import sent_tokenize
from nltk import ngrams
from collections import Counter
import networkx as nx
from textblob import TextBlob
from nltk.corpus import stopwords
from sentiment_analysis import analyze_sentiment
from similarity_analysis import compute_all_similarities


nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


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
    similarities = compute_all_similarities(documents)

    # Calculer les moyennes pour chaque méthode
    averages = {method: np.mean(similarity_matrix)
                for method, similarity_matrix in similarities.items()}

    return {
        'similarities': similarities,
        'averages': averages
    }


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


def extract_key_phrases(text, num_phrases=5):
    sentences = sent_tokenize(text)

    # Utiliser les stopwords français de NLTK
    stop_words = list(stopwords.words('french'))  # Convertir en liste

    # Utiliser TF-IDF pour identifier les phrases importantes
    vectorizer = TfidfVectorizer(stop_words=stop_words)

    # Vérifier s'il y a suffisamment de phrases pour l'analyse
    if len(sentences) < 2:
        return sentences  # Retourner toutes les phrases si moins de 2

    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Calculer l'importance de chaque phrase
    importance_scores = np.sum(tfidf_matrix.toarray(), axis=1)

    # Trier les phrases par importance
    ranked_sentences = sorted(
        ((importance_scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # Sélectionner et nettoyer les top phrases
    top_phrases = []
    for _, sentence in ranked_sentences[:num_phrases]:
        cleaned_sentence = ' '.join(sentence.split()[:20])  # Limiter à 20 mots
        cleaned_sentence = cleaned_sentence.replace('T :', '').strip()
        if cleaned_sentence not in top_phrases:
            top_phrases.append(cleaned_sentence)

    return top_phrases


def analyze_co_occurrences(documents, n_range=(2, 3), top_n=10):
    stop_words = set(stopwords.words('french'))

    # Mots-clés liés au sujet de l'étude (à ajuster selon vos besoins)
    keywords = {'cours', 'application', 'ia',
                'intelligence artificielle', 'enseignement', 'étudiant'}

    all_ngrams = []
    for doc in documents:
        words = [word.lower() for word in doc.split() if word.lower()
                 not in stop_words and len(word) > 2]
        for n in range(n_range[0], n_range[1] + 1):
            all_ngrams.extend([' '.join(gram) for gram in ngrams(
                words, n) if any(keyword in gram for keyword in keywords)])

    # Calculer TF-IDF pour les n-grammes
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([' '.join(all_ngrams)])

    # Obtenir les scores TF-IDF
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))

    # Compter les occurrences et pondérer par TF-IDF
    ngram_freq = Counter(all_ngrams)
    weighted_freq = {
        ngram: count * tfidf_scores.get(ngram, 0) for ngram, count in ngram_freq.items()}

    # Trier et prendre les top n
    top_ngrams = sorted(weighted_freq.items(),
                        key=lambda x: x[1], reverse=True)[:top_n]

    return top_ngrams  # Ceci retourne une liste de tuples (ngram, score)


def perform_advanced_analysis(file_stats, all_text, progress=None):
    # Utiliser le texte complet à l'index 1
    documents = [stat[1] for stat in file_stats]

    if progress:
        progress("Analyse de similarité des documents")
    similarity_results = analyze_document_similarity(documents)

    if progress:
        progress("Extraction des phrases clés")
    key_phrases = [extract_key_phrases(doc) for doc in documents]

    if progress:
        progress("Analyse des co-occurrences")
    co_occurrences = analyze_co_occurrences(documents)

    if progress:
        progress("Calcul de la richesse lexicale")
    lexical_richness = analyze_lexical_richness(all_text)

    if progress:
        progress("Analyse de sentiment")
    sentiments = [analyze_sentiment(doc) for doc in documents]

    if progress:
        progress("Analyse avancée terminée")

    return {
        "similarity_results": similarity_results,
        "key_phrases": key_phrases,
        "co_occurrences": co_occurrences,
        "lexical_richness": lexical_richness,
        "sentiments": sentiments
    }
