import os
import docx


def list_and_analyze_files(directory):
    print(f"Contenu du répertoire '{directory}':")
    files = os.listdir(directory)
    for file in files:
        print(f"- {file}")

    print("\nAnalyse des fichiers .docx:")
    for file in files:
        if file.endswith('.docx'):
            file_path = os.path.join(directory, file)
            try:
                doc = docx.Document(file_path)
                text = " ".join(
                    [paragraph.text for paragraph in doc.paragraphs])
                word_count = len(text.split())
                print(f"{file}: {word_count} mots")
                print(f"Premiers 50 caractères: {text[:50]}...")
            except Exception as e:
                print(f"Erreur lors de l'ouverture de {file}: {str(e)}")


# Chemin vers le dossier contenant les fichiers DOCX
data_directory = 'data'

# Vérifier si le répertoire existe
if not os.path.exists(data_directory):
    print(f"Le répertoire '{data_directory}' n'existe pas.")
else:
    list_and_analyze_files(data_directory)
