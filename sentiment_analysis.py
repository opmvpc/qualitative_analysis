from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import logging
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import numpy as np

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gestion du cache pour CamemBERT
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'model_cache')
os.makedirs(CACHE_DIR, exist_ok=True)


def textblob_analysis(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


def nltk_analysis(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']


def camembert_analysis(text, show_warnings=False):
    if not show_warnings:
        import warnings
        warnings.filterwarnings("ignore")

    try:
        # Initialiser le pipeline
        pipe = pipeline("text-classification",
                        model="ac0hik/Sentiment_Analysis_French", device="mps")

        # Tokeniser le texte en phrases
        sentences = sent_tokenize(text)

        # Analyser chaque phrase
        results = []
        for sentence in sentences:
            result = pipe(sentence)[0]

            # Convertir le label en score numérique
            score = 1 if result['label'] == 'POSITIVE' else -1

            # Multiplier par la confidence pour obtenir un score pondéré
            weighted_score = score * result['score']

            results.append(weighted_score)

        # Calculer la moyenne des scores
        average_score = np.mean(results)

        # Normaliser le score final entre -1 et 1
        final_score = np.tanh(average_score)

        return final_score

    except Exception as e:
        logger.warning(
            f"Impossible d'utiliser le modèle d'analyse de sentiment : {str(e)}")
        return None


def analyze_sentiment(text):
    results = {
        "textblob": textblob_analysis(text),
        "nltk": nltk_analysis(text),
        "camembert": camembert_analysis(text)
    }
    return results
