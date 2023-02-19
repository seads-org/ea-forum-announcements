import re
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from pandas import DataFrame

def extract_paragraphs(text:str):
    """
    Extracts paragraphs from text. Designed for texts extracted with html2text.
    Args:
        text (str): markdown-like text without emphasis, images or tables
    """
    LIST_REG = re.compile(r"^ *([*-]+|\d+\.)")
    header_level = 0
    text_parts = []
    for line in text.splitlines():
        if len(line.strip()) == 0:
            continue

        if line.startswith("#"):
            header_level = len(line) - len(line.lstrip("#"))
            line = line.lstrip("#").strip()
            text_parts.append((header_level, line, "HEAD"))
            continue

        if LIST_REG.match(line):
            offset = len(line) - len(LIST_REG.sub("", line))
            # line = LIST_REG.sub("", line).strip()
            text_parts.append((header_level + offset, line.strip(), "LIST"))
            continue

        text_parts.append((header_level, line.strip(), "TEXT"))

    text_parts = DataFrame(text_parts, columns=["level", "text", "type"])
    text_parts['n_words'] = text_parts.text.map(word_tokenize).map(len)
    return text_parts


def split_long_paragraphs(paragraphs:DataFrame, max_n_words:int):
    if paragraphs.n_words.max() <= max_n_words:
        return paragraphs

    rows = []
    for _,row in paragraphs.iterrows():
        if row.n_words < max_n_words:
            rows.append(dict(row))
            continue

        sents = sent_tokenize(row.text) # It's not perfect but shouldn't have major impact (TODO: train tokenizer first)
        rows.extend([{"level": row.level + 1, "text": s, "type": "SENTENCE", "n_words": len(word_tokenize(s))} for s in sents])
    return DataFrame(rows)


def collapse_paragraphs(level_paragraphs:DataFrame, max_n_words:int):
    texts = []
    n_words_all = []
    n_words = 0
    text = []
    for t, n in zip(level_paragraphs.text, level_paragraphs.n_words):
        if n + n_words < max_n_words:
            text.append(t)
            n_words += n
            continue

        if n_words > 0:
            if n_words > max_n_words: # last text was too long
                print(f"Warning: paragraph too long ({n_words} words): {text}")
            texts.append("\n".join(text))
            n_words_all.append(n_words)

        text = [t]
        n_words = n
        continue

    if n_words > 0:
        if n_words > max_n_words: # last text was too long
            print(f"Warning: paragraph too long ({n_words} words): {text}")
        texts.append("\n".join(text))
        n_words_all.append(n_words)
    return DataFrame({"level": level_paragraphs.level.values[0], "text": texts, "type": "COLLAPSED", "n_words":n_words_all})


def collapse_paragraphs_iteratively(paragraphs, max_n_words:int):
    # Split by sequential levels
    # TODO: Ideally, also need to split by type here
    seq_levels = paragraphs.level.diff().fillna(0).map(bool).cumsum()
    pars_per_seq_level = paragraphs.groupby(seq_levels).apply(lambda x: [x]).map(lambda x: x[0]).values

    # collapse paragraphs
    pars_collapsed = []
    for pars in pars_per_seq_level:
        pars_col = collapse_paragraphs(pars, max_n_words)
        if (pars_col.shape[0] == 1) and (len(pars_collapsed) > 0):
            pars_collapsed[-1] = pd.concat([
                pars_collapsed[-1].iloc[:-1],
                collapse_paragraphs(
                    pd.concat([pars_collapsed[-1].iloc[-1:], pars_col]),
                    max_n_words
                )
            ])
        else:
            pars_collapsed.append(pars_col)
    return pd.concat(pars_collapsed)
