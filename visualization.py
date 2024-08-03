from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.console import Group


def create_ascii_bar_chart(word_freq_data, top_n=10):
    top_words = word_freq_data[:top_n]
    max_freq = max(freq for _, freq in top_words)
    max_word_len = max(len(word) for word, _ in top_words)

    chart = "Top 10 mots les plus fréquents:\n\n"
    for word, freq in top_words:
        bar_length = int((freq / max_freq) * 20)
        chart += f"{word.ljust(max_word_len)} | {'█' * bar_length} {freq}\n"

    return chart


def create_summary_and_stats_panel(summary_data, file_stats):
    summary = Table.grid(padding=1)
    summary.add_column(style="cyan", justify="right")
    summary.add_column(style="magenta")
    summary.add_row("Fichiers:", str(summary_data["file_count"]))
    summary.add_row("Total mots:", str(summary_data["total_words"]))
    summary.add_row("Mots uniques:", str(summary_data["unique_words"]))
    summary.add_row("TTR:", f"{summary_data['ttr']:.2%}")
    summary.add_row("Sentiment:", f"{summary_data['sentiment']:.2f}")

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


def create_bigram_table(bigram_data):
    table = Table(title="Top 10 Bi-grammes")
    table.add_column("Bi-gramme", style="cyan")
    table.add_column("Fréquence", justify="right")

    for bigram, freq in bigram_data:
        table.add_row(" ".join(bigram), str(freq))

    return table


def create_advanced_analysis_panels(advanced_results):
    similarity_table = Table(title="Similarité entre documents")
    similarity_table.add_column("Document", style="cyan")
    for i in range(len(advanced_results["similarity_matrix"])):
        similarity_table.add_column(f"Doc {i+1}", justify="right")

    for i, row in enumerate(advanced_results["similarity_matrix"]):
        similarity_table.add_row(f"Doc {i+1}", *[f"{val:.2f}" for val in row])

    topics_table = Table(title="Thèmes principaux")
    topics_table.add_column("Thème", style="cyan")
    topics_table.add_column("Mots-clés", style="magenta")
    for i, topic in enumerate(advanced_results["topics"]):
        topics_table.add_row(f"Thème {i+1}", topic)

    lexical_table = Table(title="Richesse lexicale")
    lexical_table.add_column("Mesure", style="cyan")
    lexical_table.add_column("Valeur", justify="right")
    for measure, value in advanced_results["lexical_richness"].items():
        lexical_table.add_row(measure, f"{value:.2f}")

    phrases_table = Table(title="Phrases clés")
    phrases_table.add_column("Phrase", style="cyan")
    phrases_table.add_column("Fréquence", justify="right")
    for phrase, freq in advanced_results["key_phrases"]:
        phrases_table.add_row(phrase, str(freq))

    return Group(
        Panel(similarity_table, title="Similarité entre documents",
              border_style="blue"),
        Panel(topics_table, title="Analyse thématique", border_style="green"),
        Panel(lexical_table, title="Richesse lexicale", border_style="yellow"),
        Panel(phrases_table, title="Phrases clés", border_style="red")
    )


def create_header(text):
    return Text(text, style="bold magenta", justify="center")


def create_summary_panel(summary_data):
    summary = Table.grid(padding=1)
    summary.add_column(style="cyan", justify="right")
    summary.add_column(style="magenta")
    summary.add_row("Fichiers:", str(summary_data["file_count"]))
    summary.add_row("Total mots:", str(summary_data["total_words"]))
    summary.add_row("Mots uniques:", str(summary_data["unique_words"]))
    summary.add_row("TTR:", f"{summary_data['ttr']:.2%}")
    summary.add_row("Sentiment:", f"{summary_data['sentiment']:.2f}")
    return Panel(summary, title="Résumé de l'analyse", border_style="blue")


def create_word_frequency_table(word_freq_data):
    table = Table(title="Les 50 mots les plus fréquents", show_lines=True)
    cols = 5
    rows = 10

    for i in range(cols):
        table.add_column(f"Mot", style="magenta", width=12)
        table.add_column(f"Fréq", justify="right", width=6)

    for i in range(rows):
        row_data = []
        for j in range(cols):
            idx = i + j * rows
            if idx < len(word_freq_data):
                word, freq = word_freq_data[idx]
                row_data.extend([word[:10], str(freq)])
            else:
                row_data.extend(["", ""])
        table.add_row(*row_data)

    return table


def create_similarity_matrix(similarity_matrix):
    table = Table(title="Similarité entre documents")
    table.add_column("Document", style="cyan")
    for i in range(len(similarity_matrix)):
        table.add_column(f"Doc {i+1}", justify="right")

    for i, row in enumerate(similarity_matrix):
        table.add_row(f"Doc {i+1}", *[f"{val:.2f}" for val in row])

    return table


def create_topics_table(topics):
    table = Table(title="Thèmes principaux")
    table.add_column("Thème", style="cyan")
    table.add_column("Mots-clés", style="magenta")
    for i, topic in enumerate(topics):
        table.add_row(f"Thème {i+1}", topic)
    return table


def create_lexical_richness_table(lexical_richness):
    table = Table(title="Richesse lexicale")
    table.add_column("Mesure", style="cyan")
    table.add_column("Valeur", justify="right")
    for measure, value in lexical_richness.items():
        table.add_row(measure, f"{value:.2f}")
    return table


def create_layout(basic_results, advanced_results):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body")
    )
    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=2)
    )
    layout["right"].split_column(
        Layout(name="top_right"),
        Layout(name="bottom_right")
    )

    layout["header"].update(create_header(
        "Analyse Qualitative des Entretiens"))
    layout["left"].update(Group(
        create_summary_panel(basic_results["summary"]),
        Panel(create_similarity_matrix(
            advanced_results["similarity_matrix"]), title="Similarité entre documents", border_style="green"),
        Panel(create_lexical_richness_table(
            advanced_results["lexical_richness"]), title="Richesse lexicale", border_style="yellow")
    ))
    layout["top_right"].update(create_word_frequency_table(
        basic_results["word_frequencies"]))
    layout["bottom_right"].update(Panel(create_topics_table(
        advanced_results["topics"]), title="Analyse thématique", border_style="red"))

    return layout
