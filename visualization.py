from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def create_header(text):
    return Text(text, style="bold magenta", justify="center")


def create_summary_panel(summary_data, file_stats):
    summary = Table.grid(padding=1)
    summary.add_column(style="cyan", justify="right")
    summary.add_column(style="magenta")
    for key, value in summary_data.items():
        summary.add_row(f"{key}:", f"{value:.2f}" if isinstance(
            value, float) else str(value))

    # Ajout du tableau des statistiques par document
    doc_stats_table = Table(title="Statistiques par document")
    doc_stats_table.add_column("Document", style="cyan")
    doc_stats_table.add_column("Mots", justify="right")
    doc_stats_table.add_column("Phrases", justify="right")
    doc_stats_table.add_column("Mots/Phrase", justify="right")

    for filename, _, word_count, sentence_count, avg_sentence_length in file_stats:
        doc_stats_table.add_row(
            filename,
            str(word_count),
            str(sentence_count),
            f"{avg_sentence_length:.2f}"
        )

    # Utilisation de Group pour combiner les deux tables
    content = Group(
        Text("Résumé global", style="bold"),
        summary,
        Text("\nStatistiques par document", style="bold"),
        doc_stats_table
    )

    return Panel(content, title="Résumé de l'analyse", border_style="blue")


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

    return Panel(table, title="Fréquence des mots", border_style="red")


def create_similarity_tables(similarity_results):
    grid = Table.grid(expand=True)
    grid.add_column("Left", ratio=1)
    grid.add_column("Right", ratio=1)

    methods = list(similarity_results['similarities'].keys())
    for i in range(0, len(methods), 2):
        left_method = methods[i]
        right_method = methods[i+1] if i+1 < len(methods) else None

        left_table = create_single_similarity_table(
            left_method, similarity_results['similarities'][left_method])
        if right_method:
            right_table = create_single_similarity_table(
                right_method, similarity_results['similarities'][right_method])
            grid.add_row(Panel(left_table, border_style="green"),
                         Panel(right_table, border_style="green"))
        else:
            grid.add_row(Panel(left_table, border_style="green"), "")

    return Panel(grid, title="Similarités", border_style="blue")


def create_single_similarity_table(method, similarity_matrix):
    table = Table(title=f"Similarité - {method.upper()}")
    table.add_column("Document", style="cyan")
    for i in range(len(similarity_matrix)):
        table.add_column(f"Doc {i+1}", justify="right")
    for i, row in enumerate(similarity_matrix):
        table.add_row(f"Doc {i+1}", *[f"{val:.2f}" for val in row])
    return table


def create_sentiment_table(sentiments):
    table = Table(title="Analyse comparative de sentiment")
    table.add_column("Document", style="cyan")
    table.add_column("TextBlob", justify="right")
    table.add_column("NLTK", justify="right")
    if any('camembert' in s for s in sentiments):
        table.add_column("CamemBERT", justify="right")
    for i, sentiment in enumerate(sentiments):
        row = [f"Doc {i+1}", f"{sentiment['textblob']:.2f}",
               f"{sentiment['nltk']:.2f}"]
        if 'camembert' in sentiment:
            row.append(f"{sentiment['camembert']:.2f}")
        table.add_row(*row)
    return Panel(table, title="Analyse de sentiment", border_style="yellow")


def create_lexical_richness_table(lexical_richness):
    table = Table(title="Richesse lexicale")
    table.add_column("Mesure", style="cyan")
    table.add_column("Valeur", justify="right")
    for measure, value in lexical_richness.items():
        table.add_row(measure, f"{value:.2f}")
    return Panel(table, border_style="blue")


def create_key_phrases_table(key_phrases):
    table = Table(title="Phrases-clés par document")
    table.add_column("Document", style="cyan")
    table.add_column("Phrases-clés", style="magenta", no_wrap=True)
    for i, phrases in enumerate(key_phrases):
        table.add_row(f"Doc {i+1}", "\n".join(phrases))
    return Panel(table, title="Phrases-clés par document", border_style="red")


def create_co_occurrences_table(co_occurrences):
    table = Table(title="Co-occurrences fréquentes")
    table.add_column("Mots", style="cyan")
    table.add_column("Score", justify="right")
    for ngram, score in co_occurrences:
        table.add_row(ngram, f"{score:.4f}")
    return Panel(table, title="Co-occurrences fréquentes", border_style="green")


def create_similarity_average_table(averages):
    table = Table(title="Moyennes de similarité par méthode")
    table.add_column("Méthode", style="cyan")
    table.add_column("Moyenne", justify="right")
    for method, avg in averages.items():
        table.add_row(method.upper(), f"{avg:.2f}")
    return Panel(table, title="Moyennes de similarité", border_style="blue")


def display_results(basic_results, advanced_results):
    console.print(create_header("Analyse Qualitative des Entretiens"))
    console.print(create_summary_panel(
        basic_results["summary"], basic_results["file_stats"]))
    console.print(create_word_frequency_table(
        basic_results["word_frequencies"]))
    console.print(create_similarity_tables(
        advanced_results["similarity_results"]))
    console.print(create_similarity_average_table(
        advanced_results["similarity_results"]["averages"]))
    console.print(create_sentiment_table(advanced_results["sentiments"]))
    console.print(create_lexical_richness_table(
        advanced_results["lexical_richness"]))
    console.print(create_key_phrases_table(advanced_results["key_phrases"]))
    console.print(create_co_occurrences_table(
        advanced_results["co_occurrences"]))


def run_analysis_with_progress(data_processing_func, basic_analysis_func, advanced_analysis_func, data_directory):
    with console.status("[bold green]Analyse en cours...") as status:
        status.update(status="Lecture et prétraitement des fichiers...")
        word_frequencies, file_stats, all_text = data_processing_func(
            data_directory)

        status.update(status="Exécution de l'analyse de base...")
        basic_results = basic_analysis_func(
            word_frequencies, file_stats, all_text)

        status.update(status="Exécution de l'analyse avancée...")
        advanced_results = advanced_analysis_func(file_stats, all_text)

        status.update(status="Affichage des résultats...")
        display_results(basic_results, advanced_results)

    console.print(
        "[bold green]Analyse terminée. Utilisez la barre de défilement pour voir tous les résultats.")
