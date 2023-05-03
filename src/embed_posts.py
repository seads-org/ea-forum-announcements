import argparse
import math

import nltk
import numpy as np
import pandas as pd
import torch.nn.functional as F
from pandas import DataFrame
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

from constants import RAW_DATA_PATH, SBERT_PATH, MAX_TOKENS, EMBEDDING_SIZE
from src.text_split import extract_paragraphs, split_long_paragraphs, collapse_paragraphs_iteratively
from src.paths import datap


def get_paragraph_split(data):
    """Split posts to paragraphs of max 400 words"""
    data['paragraphs'] = data.body.progress_map(extract_paragraphs)
    data['paragraphs'] = data.paragraphs.progress_map(lambda p: split_long_paragraphs(p, max_n_words=MAX_TOKENS))
    data = data[~data.apply(lambda x: x.paragraphs.empty, axis=1)]
    data['paragraphs_split'] = data.paragraphs.progress_map(
        lambda x: collapse_paragraphs_iteratively(x, max_n_words=MAX_TOKENS))
    paragraph_split = pd.concat([
        DataFrame({"postId": r._id, "text": r.paragraphs_split.text.values, "label": r.label})
        for pid, r in data.iterrows()
    ], ignore_index=True)
    return paragraph_split


def embed_e5base(paragraph_split, model, tokenizer, batch_size):
    """
    Embed posts' paragraphs using e5-base model
    Each input text should start with "query: " or "passage: ".
    For tasks other than retrieval, you can simply use the "query: " prefix.
    """
    paragraph_split.text = paragraph_split.text.apply(lambda x: "query: " + x)
    n_batches = math.ceil(len(paragraph_split) / batch_size)
    paragraphs_embeddings = np.zeros((n_batches, EMBEDDING_SIZE))
    for idx in tqdm(range(n_batches)):
        idx1, idx2 = idx * batch_size, (idx + 1) * batch_size
        idx2 = min(len(paragraph_split), idx2)
        batch_dict = tokenizer(paragraph_split.text[idx1:idx2].tolist(), max_length=MAX_TOKENS, padding=True,
                               truncation=True, return_tensors='pt')
        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        paragraphs_embeddings[idx, :] = F.normalize(embeddings, p=2, dim=1).detach()
    return paragraphs_embeddings


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pooling for e5-base embeddings"""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Generate dataset',
        description='Split posts by paragraph of max 400 words and embed them'
                    ' using either ea-forum trained SBERT or e5-base model from hugging face')

    parser.add_argument('--filename', default="labeled_posts_embedded.csv",
                        help="Filename of the .csv file with embeddings for posts")
    parser.add_argument('--embeddings', default="ea-forum",
                        help="Name of embedding type, available: 'ea-forum' or 'e5-base'")

    args = parser.parse_args()

    nltk.download('punkt')
    tqdm.pandas()
    data = load_from_disk(RAW_DATA_PATH).to_pandas()
    paragraph_split = get_paragraph_split(data)
    batch_size = 1 # memory
    if args.embeddings == "ea-forum":
        model = SentenceTransformer.load(SBERT_PATH)
        paragraphs_embeddings = model.encode(paragraph_split.text, batch_size=batch_size, show_progress_bar=True)
    elif args.embeddings == "e5-base":
        model = AutoModel.from_pretrained('intfloat/e5-base')
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base')
        paragraphs_embeddings = embed_e5base(paragraph_split, model, tokenizer, batch_size)
    else:
        raise ValueError(f"Unknown embedding type: {args.embeddings}, available: ea-forum or e5-base")

    paragraph_split = pd.concat((paragraph_split, pd.DataFrame(paragraphs_embeddings)), axis=1)
    dataset = paragraph_split.groupby("postId").mean(numeric_only=True)
    dataset.label = dataset.label.astype("int8")
    dataset = dataset.sample(frac=1)
    dataset.to_csv(datap(args.filename))
