import os
from collections import Counter
import docx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.console import Group
from custom_stopwords import CUSTOM_STOPWORDS
from textblob import TextBlob

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

console = Console()


def read_docx(file_path):
    doc = docx.Document(file_path)
    return " ".join([paragraph.text for paragraph in doc.paragraphs])


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('french')).union(CUSTOM_STOPWORDS)
    return [word for word in tokens if word.isalpha() and len(word) > 2 and word not in stop_words]


def analyze_files(directory):
    all_words = []
    file_stats = []
    all_text = ""
    for filename in os.listdir(directory):
        if filename.endswith('.docx'):
            file_path = os.path.join(directory, filename)
            text = read_docx(file_path)
            all_text += text + " "
            words = preprocess_text(text)
            all_words.extend(words)
            sentences = sent_tokenize(text)
            avg_sentence_length = sum(len(sent.split())
                                      for sent in sentences) / len(sentences)
            file_stats.append((filename, len(text), len(
                words), len(sentences), avg_sentence_length))

    word_freq = Counter(all_words)
    return word_freq, file_stats, all_text


def create_ascii_bar_chart(word_freq, top_n=10):
    top_words = word_freq.most_common(top_n)
    max_freq = max(freq for _, freq in top_words)
    max_word_len = max(len(word) for word, _ in top_words)

    chart = "Top 10 mots les plus fréquents:\n\n"
    for word, freq in top_words:
        bar_length = int((freq / max_freq) * 20)
        chart += f"{word.ljust(max_word_len)} | {'█' * bar_length} {freq}\n"

    return chart


def create_word_frequency_table(word_freq, top_n=50):
    table = Table(title="Les 50 mots les plus fréquents", show_lines=True)

    cols = 10
    rows = -(-top_n // cols)  # Ceiling division

    for i in range(cols):
        table.add_column(f"Mot", style="magenta", width=8)
        table.add_column(f"F", justify="right", width=4)

    for i in range(rows):
        row_data = []
        for j in range(cols):
            idx = i + j * rows
            if idx < top_n:
                word, freq = word_freq.most_common(top_n)[idx]
                row_data.extend([word[:7], str(freq)])
            else:
                row_data.extend(["", ""])
        table.add_row(*row_data)

    return table


def create_summary_and_stats(file_stats, word_freq, all_text):
    total_words = sum(stat[1] for stat in file_stats)
    unique_words = len(word_freq)
    ttr = unique_words / total_words if total_words > 0 else 0
    blob = TextBlob(all_text)
    sentiment = blob.sentiment.polarity

    summary = Table.grid(padding=1)
    summary.add_column(style="cyan", justify="right")
    summary.add_column(style="magenta")
    summary.add_row("Fichiers:", str(len(file_stats)))
    summary.add_row("Total mots:", str(total_words))
    summary.add_row("Mots uniques:", str(unique_words))
    summary.add_row("TTR:", f"{ttr:.2%}")
    summary.add_row("Sentiment:", f"{sentiment:.2f}")

    stats_table = Table(title="Statistiques des fichiers",
                        show_header=True, header_style="bold cyan")
    stats_table.add_column("Fichier", style="cyan")
    stats_table.add_column("Mots", justify="right")
    stats_table.add_column("Uniq", justify="right")
    stats_table.add_column("Phr", justify="right")
    stats_table.add_column("Moy/Phr", justify="right")

    for filename, text_length, words_count, sent_count, avg_sent_len in file_stats:
        stats_table.add_row(
            filename[:10],
            str(text_length),
            str(words_count),
            str(sent_count),
            f"{avg_sent_len:.1f}"
        )

    return Group(
        Panel(summary, title="Résumé de l'analyse", border_style="blue"),
        Panel(stats_table, title="Statistiques détaillées", border_style="green")
    )


def get_top_bigrams(text, n=10):
    words = preprocess_text(text)
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(words)
    # Filtre les bi-grammes apparaissant au moins 2 fois
    finder.apply_freq_filter(2)
    return finder.nbest(bigram_measures.pmi, n)


def create_bigram_table(all_text):
    top_bigrams = get_top_bigrams(all_text)
    table = Table(title="Top 10 Bi-grammes")
    table.add_column("Bi-gramme", style="cyan")
    table.add_column("Fréquence", justify="right")

    for bigram in top_bigrams:
        freq = all_text.lower().count(" ".join(bigram))
        if freq > 0:  # N'affiche que les bi-grammes avec une fréquence > 0
            table.add_row(" ".join(bigram), str(freq))

    return table


def create_layout(file_stats, word_freq, all_text):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="top", size=15),
        Layout(name="middle"),
        Layout(name="bottom", size=15)
    )

    layout["top"].split_row(
        Layout(name="summary"),
        Layout(name="stats")
    )

    layout["bottom"].split_row(
        Layout(name="graph"),
        Layout(name="bigrams")
    )

    layout["header"].update(Text(
        "Analyse Qualitative des Entretiens", style="bold magenta", justify="center"))
    summary_and_stats = create_summary_and_stats(
        file_stats, word_freq, all_text)
    layout["top"]["summary"].update(summary_and_stats.renderables[0])
    layout["top"]["stats"].update(summary_and_stats.renderables[1])
    layout["middle"].update(create_word_frequency_table(word_freq, 50))
    layout["bottom"]["graph"].update(Panel(create_ascii_bar_chart(
        word_freq), title="Top 10 mots les plus fréquents"))
    layout["bottom"]["bigrams"].update(create_bigram_table(all_text))

    return layout


def main():
    data_directory = 'data'
    word_frequencies, file_stats, all_text = analyze_files(data_directory)

    layout = create_layout(file_stats, word_frequencies, all_text)
    console.print(layout)


if __name__ == "__main__":
    main()
