from textblob import TextBlob
from rich.progress import Progress


def create_word_frequency_data(word_freq, top_n=50):
    return word_freq.most_common(top_n)


def create_summary_data(file_stats, word_freq, all_text):
    # Utilise l'index 2 qui contient le nombre de mots
    total_words = sum(stat[2] for stat in file_stats)
    unique_words = len(word_freq)
    ttr = unique_words / total_words if total_words > 0 else 0
    blob = TextBlob(all_text)
    sentiment = blob.sentiment.polarity
    return {
        "file_count": len(file_stats),
        "total_words": total_words,
        "unique_words": unique_words,
        "ttr": ttr,
        "sentiment": sentiment
    }


def perform_basic_analysis(word_frequencies, file_stats, all_text, progress=None):
    word_freq_data = create_word_frequency_data(word_frequencies)
    if progress:
        progress("Calcul des fréquences de mots terminé")

    summary_data = create_summary_data(file_stats, word_frequencies, all_text)
    if progress:
        progress("Calcul des statistiques sommaires terminé")

    if progress:
        progress("Analyse de base terminée")

    return {
        "word_frequencies": word_freq_data,
        "summary": summary_data,
        "file_stats": file_stats
    }
