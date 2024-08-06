import os
from collections import Counter
import docx
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import re
import logging
from rich.logging import RichHandler
from custom_stopwords import CUSTOM_STOPWORDS

# Configuration du logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("rich")

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


def read_docx(file_path):
    """Lit un fichier .docx et retourne son contenu sous forme de texte."""
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])


def remove_interviewer_prefix(text):
    """Supprime le préfixe 'T:' des questions de l'interviewer."""
    return re.sub(r'(?m)^T\s*:\s*', '', text)


def remove_interviewee_names(text):
    """Supprime les noms des interviewés au début des lignes."""
    names_to_remove = ['Roman Couchard', 'Catherine Franken', 'Sylvain Léonis']
    pattern = '|'.join(map(re.escape, names_to_remove))
    return re.sub(f'(?m)^({pattern})\\s*:', '', text)


def preprocess_text(text):
    """Effectue un prétraitement minimal du texte."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def process_single_file(file_path):
    """Traite un seul fichier d'entretien."""
    log.info(f"Traitement du fichier: {file_path}")

    text = read_docx(file_path)
    original_length = len(text)
    log.info(f"Longueur originale du texte: {original_length} caractères")

    text = remove_interviewer_prefix(text)
    text = remove_interviewee_names(text)
    text = preprocess_text(text)

    processed_length = len(text)
    log.info(
        f"Longueur du texte après traitement: {processed_length} caractères")

    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)

    avg_sentence_length = sum(len(sent.split())
                              for sent in sentences) / len(sentences) if sentences else 0
    log.info(
        f"Statistiques: {len(words)} mots, {len(sentences)} phrases, longueur moyenne de phrase: {avg_sentence_length:.2f}")

    return text, words, len(words), len(sentences), avg_sentence_length


def get_word_frequencies(words):
    """Calcule la fréquence des mots en appliquant le filtrage."""
    stop_words = set(nltk.corpus.stopwords.words(
        'french')).union(CUSTOM_STOPWORDS)
    filtered_words = [word for word in words if word.isalpha() and len(
        word) > 2 and word not in stop_words]
    return Counter(filtered_words)


def read_and_preprocess_files(directory, progress=None):
    """Lit et prétraite tous les fichiers d'entretien dans le répertoire donné."""
    all_words = []
    file_stats = []
    all_text = ""

    files = [f for f in os.listdir(directory) if f.endswith('.docx')]
    total_files = len(files)

    for i, filename in enumerate(files):
        if progress:
            progress(f"Traitement du fichier {i+1}/{total_files}: {filename}")

        file_path = os.path.join(directory, filename)
        text, words, word_count, sentence_count, avg_sentence_length = process_single_file(
            file_path)

        all_text += text + "\n\n"  # Ajoute deux sauts de ligne entre les entretiens
        all_words.extend(words)
        file_stats.append((filename, text, word_count,
                          sentence_count, avg_sentence_length))

    if progress:
        progress("Calcul des fréquences de mots...")

    word_freq = get_word_frequencies(all_words)

    log.info(
        f"Traitement terminé. {len(files)} fichiers traités, {len(all_words)} mots au total")
    return word_freq, file_stats, all_text
