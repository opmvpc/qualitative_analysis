from data_processing import read_and_preprocess_files
from basic_analysis import perform_basic_analysis
from advanced_analysis import perform_advanced_analysis
from visualization import create_layout
from rich.console import Console


def main():
    data_directory = 'data'
    word_frequencies, file_stats, all_text = read_and_preprocess_files(
        data_directory)

    basic_results = perform_basic_analysis(
        word_frequencies, file_stats, all_text)
    advanced_results = perform_advanced_analysis(file_stats, all_text)

    layout = create_layout(basic_results, advanced_results)

    console = Console()
    console.print(layout)


if __name__ == "__main__":
    main()
