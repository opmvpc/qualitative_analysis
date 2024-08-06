from data_processing import read_and_preprocess_files
from basic_analysis import perform_basic_analysis
from advanced_analysis import perform_advanced_analysis
from visualization import run_analysis_with_progress


def main():
    data_directory = 'data'

    run_analysis_with_progress(
        read_and_preprocess_files,
        perform_basic_analysis,
        perform_advanced_analysis,
        data_directory
    )


if __name__ == "__main__":
    main()
