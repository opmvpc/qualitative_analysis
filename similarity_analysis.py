import logging
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CamembertTokenizer, CamembertModel
import torch
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
import json
import hashlib
import tiktoken
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


load_dotenv()
# Configuration OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration CamemBERT
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertModel.from_pretrained("camembert-base")


def chunk_text(text, encoding_name, max_tokens):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(encoding.decode(chunk))
    return chunks


def compute_document_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def save_embeddings(embeddings, file_path='embeddings_cache.json'):
    with open(file_path, 'w') as f:
        json.dump(embeddings, f)


def load_embeddings(file_path='embeddings_cache.json'):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def get_openai_embedding(text, model="text-embedding-3-small", max_tokens=8191):
    chunks = chunk_text(text, "cl100k_base", max_tokens)
    logger.debug(f"Texte divisé en {len(chunks)} chunks")

    embeddings_cache = load_embeddings()
    doc_hash = compute_document_hash(text)

    if doc_hash in embeddings_cache:
        logger.debug("Embedding trouvé dans le cache")
        return np.array(embeddings_cache[doc_hash])

    logger.debug("Calcul des embeddings pour chaque chunk")
    chunk_embeddings = []
    for i, chunk in enumerate(chunks):
        logger.debug(f"Calcul de l'embedding pour le chunk {i+1}")
        response = client.embeddings.create(input=[chunk], model=model)
        chunk_embeddings.append(response.data[0].embedding)

    logger.debug("Calcul de l'embedding moyen")
    avg_embedding = np.mean(chunk_embeddings, axis=0)
    avg_embedding = avg_embedding / \
        np.linalg.norm(avg_embedding)  # Normalisation

    logger.debug(f"Taille de l'embedding moyen : {len(avg_embedding)}")
    logger.debug("Sauvegarde de l'embedding dans le cache")
    embeddings_cache[doc_hash] = avg_embedding.tolist()
    save_embeddings(embeddings_cache)

    return avg_embedding


def compute_openai_similarity(texts):
    try:
        logger.debug(
            f"Début du calcul de similarité OpenAI pour {len(texts)} textes")
        embeddings = []
        for i, text in enumerate(texts):
            logger.debug(f"Calcul de l'embedding pour le texte {i+1}")
            embedding = get_openai_embedding(text)
            logger.debug(
                f"Embedding calculé pour le texte {i+1}, type: {type(embedding)}, shape: {embedding.shape}")
            embeddings.append(embedding)

        logger.debug(
            f"Tous les embeddings calculés. Nombre d'embeddings: {len(embeddings)}")

        logger.debug("Conversion des embeddings en tableau numpy")
        embeddings_array = np.array(embeddings)
        logger.debug(f"Tableau numpy créé. Shape: {embeddings_array.shape}")

        logger.debug("Calcul de la similarité cosinus")
        similarity_matrix = cosine_similarity(embeddings_array)
        logger.debug(
            f"Matrice de similarité calculée, shape: {similarity_matrix.shape}")

        return similarity_matrix
    except Exception as e:
        logger.error(
            f"Erreur lors du calcul de la similarité OpenAI: {str(e)}")
        logger.exception("Traceback complet:")
        raise


def get_bert_embedding(text, max_tokens=512):
    chunks = chunk_text(text, "cl100k_base", max_tokens)
    chunk_embeddings = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt",
                           truncation=True, max_length=max_tokens, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        chunk_embedding = outputs.last_hidden_state.mean(
            dim=1).squeeze().numpy()
        chunk_embeddings.append(chunk_embedding)

    avg_embedding = np.mean(chunk_embeddings, axis=0)
    avg_embedding = avg_embedding / \
        np.linalg.norm(avg_embedding)  # Normalisation
    return avg_embedding


def compute_bert_similarity(texts):
    try:
        logger.debug(
            f"Début du calcul de similarité BERT pour {len(texts)} textes")
        embeddings = []
        for i, text in enumerate(texts):
            logger.debug(f"Calcul de l'embedding pour le texte {i+1}")
            embedding = get_bert_embedding(text)
            logger.debug(
                f"Embedding calculé pour le texte {i+1}, type: {type(embedding)}, shape: {embedding.shape}")
            embeddings.append(embedding)

        logger.debug(
            f"Tous les embeddings calculés. Nombre d'embeddings: {len(embeddings)}")

        logger.debug("Conversion des embeddings en tableau numpy")
        embeddings_array = np.array(embeddings)
        logger.debug(f"Tableau numpy créé. Shape: {embeddings_array.shape}")

        logger.debug("Calcul de la similarité cosinus")
        similarity_matrix = cosine_similarity(embeddings_array)
        logger.debug(
            f"Matrice de similarité calculée, shape: {similarity_matrix.shape}")

        return similarity_matrix
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la similarité BERT: {str(e)}")
        logger.exception("Traceback complet:")
        raise


def compute_tfidf_similarity(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


def compute_doc2vec_similarity(documents):
    tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[
                                  str(i)]) for i, doc in enumerate(documents)]
    model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count,
                epochs=model.epochs)

    vectors = [model.infer_vector(word_tokenize(doc.lower()))
               for doc in documents]
    return cosine_similarity(vectors)


def compute_fasttext_similarity(documents):
    tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
    model = FastText(tokenized_docs, vector_size=100,
                     window=5, min_count=1, workers=4)
    doc_vectors = [np.mean(
        [model.wv[word] for word in doc if word in model.wv], axis=0) for doc in tokenized_docs]
    return cosine_similarity(doc_vectors)


def compute_all_similarities(texts):
    methods = ['openai',
               # 'bert',
               'tfidf', 'doc2vec', 'fasttext']
    results = {}
    for method in methods:
        if method == 'openai':
            results[method] = compute_openai_similarity(texts)
        elif method == 'bert':
            results[method] = compute_bert_similarity(texts)
        elif method == 'tfidf':
            results[method] = compute_tfidf_similarity(texts)
        elif method == 'doc2vec':
            results[method] = compute_doc2vec_similarity(texts)
        elif method == 'fasttext':
            results[method] = compute_fasttext_similarity(texts)
    return results
