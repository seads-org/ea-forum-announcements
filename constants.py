from src.paths import datap, modelp

RAW_DATA_PATH = datap("labeled_posts")
SBERT_PATH = modelp("sbert_v1/sbert:v1")
N_EPOCHS = 10
MAX_TOKENS = 400
EMBEDDING_SIZE = 768
N_CLASSES = 3
LABEL_MAP = {"organization": 0, "project": 1, "other": 2}
