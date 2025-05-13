import os, pickle, faiss, threading, requests
from urllib.parse import quote
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
try:
    from konlpy.tag import Okt
    okt = Okt()
except:
    okt = None
import numpy as np
from sentence_transformers.util import cos_sim
import torch

# --- Keyphrase extraction (KeyBERT) ---
from keybert import KeyBERT
_KW_MODEL = KeyBERT("jhgan/ko-sbert-nli")

def _extract_keyphrases(query: str, top_n: int = 2) -> list[str]:
    candidates = _KW_MODEL.extract_keywords(
        query,
        keyphrase_ngram_range=(1, 2),
        stop_words="korean",
        top_n=top_n
    )
    return [kw for kw, _ in candidates]

# --- SBERT & DB settings ---
_MODEL_NAME  = "jhgan/ko-sbert-nli"
_DB_TXT      = "bias_db.txt"
_IDX_FILE    = "faiss_bias.index"
_SENT_FILE   = "bias_sent.pkl"

# --- Cross-Encoder ensemble ---
_CE_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/stsb-distilroberta-base",
    "monologg/koelectra-base-v3-discriminator"
]
_CE_ENSEMBLE = [CrossEncoder(m) for m in _CE_MODELS]

# --- BM25 initialization ---
_BM25 = None
_BM25_SENTS = None

def _init_bm25(db_txt: str = _DB_TXT):
    global _BM25, _BM25_SENTS
    if _BM25 is None:
        sents = [ln.strip() for ln in open(db_txt, encoding="utf-8") if ln.strip()]
        tokenized = [okt.morphs(sent) if okt else sent.split() for sent in sents]
        _BM25 = BM25Okapi(tokenized)
        _BM25_SENTS = sents
    return _BM25, _BM25_SENTS

# --- SBERT loader (singleton) ---
_SBERT_MODEL = None
_SBERT_LOCK  = threading.Lock()

def _get_sbert():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        with _SBERT_LOCK:
            if _SBERT_MODEL is None:
                print("🔧  SBERT 로딩…")
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                _SBERT_MODEL = SentenceTransformer(_MODEL_NAME, device=device)
                _SBERT_MODEL.encode(["warm-up"], convert_to_tensor=True)
                print("✅  SBERT 로딩 완료")
    return _SBERT_MODEL

# --- Cross-Encoder re-ranking ---
def rerank_with_cross_encoder(query: str, docs: list[str]) -> list[str]:
    pairs = [[query, d] for d in docs]
    scores = [ce.predict(pairs) for ce in _CE_ENSEMBLE]
    avg = np.mean(scores, axis=0)
    return [doc for _, doc in sorted(zip(avg, docs), key=lambda x: x[0], reverse=True)]

# --- MMR selection ---
def mmr(query: str, docs: list[str], k: int, lambda_param: float) -> list[str]:
    sbert = _get_sbert()
    q_emb = sbert.encode([query], convert_to_tensor=True)
    d_embs = sbert.encode(docs, convert_to_tensor=True)
    rel_scores = cos_sim(q_emb, d_embs)[0].cpu().tolist()
    selected, sel_idx, candidates = [], [], list(range(len(docs)))
    for _ in range(min(k, len(candidates))):
        if not selected:
            idx = int(np.argmax(rel_scores))
        else:
            mmr_val = []
            for i in candidates:
                div = max(cos_sim(d_embs[i].unsqueeze(0), d_embs[sel_idx]).cpu().tolist()[0])
                mmr_val.append((i, lambda_param * rel_scores[i] - (1 - lambda_param) * div))
            idx = max(mmr_val, key=lambda x: x[1])[0]
        selected.append(docs[idx]); sel_idx.append(idx); candidates.remove(idx)
    return selected

# --- Build DB FAISS index ---
def build_db_index(db_txt: str = _DB_TXT):
    sents = [ln.strip() for ln in open(db_txt, encoding="utf-8") if ln.strip()]
    embs = _get_sbert().encode(sents, convert_to_tensor=True, normalize_embeddings=True).cpu().numpy()
    idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs)
    faiss.write_index(idx, _IDX_FILE)
    pickle.dump(sents, open(_SENT_FILE, "wb"))
    print(f"✅  DB indexed: {len(sents)} sentences")

# --- Build Wiki FAISS index (once) ---
_WIKI_IDX = "faiss_wiki.index"
def build_wiki_index(wiki_txt: str):
    wiki_sents = [ln.strip() for ln in open(wiki_txt, encoding="utf-8") if ln.strip()]
    embs = _get_sbert().encode(wiki_sents, convert_to_tensor=True, normalize_embeddings=True).cpu().numpy()
    idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs)
    faiss.write_index(idx, _WIKI_IDX)
    pickle.dump(wiki_sents, open(_WIKI_SENTS_FILE, "wb"))
    print(f"✅  Wiki indexed: {len(wiki_sents)} sentences")

# --- Retrieve Wiki by embedding ---
def get_wiki_by_embedding(query: str, k: int = 2) -> list[str]:
    if not os.path.exists(_WIKI_IDX):
        raise RuntimeError("Wiki index missing; run build_wiki_index() first")
    idx = faiss.read_index(_WIKI_IDX)
    wiki_sents = pickle.load(open(_WIKI_SENTS_FILE, "rb"))
    qvec = _get_sbert().encode([query], convert_to_tensor=True, normalize_embeddings=True).cpu().numpy()
    _, I = idx.search(qvec, k)
    return [wiki_sents[i] for i in I[0]]

# --- Unified retrieval interface ---
def get_relevant(query: str, k: int=2, method: str="all", mmr_lambda: float=0.9) -> list[str]:
    candidates = []
    key_phrases = _extract_keyphrases(query, top_n=2)
    # BM25
    if method in ("bm25", "all"):
        bm25, sents = _init_bm25()
        tokens = okt.morphs(query) if okt else query.split()
        scores = bm25.get_scores(tokens)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        candidates += [sents[i] for i in top]
    # SBERT FAISS
    if method in ("sbert", "all"):
        if not os.path.exists(_IDX_FILE): build_db_index()
        idx = faiss.read_index(_IDX_FILE); sents = pickle.load(open(_SENT_FILE, "rb"))
        qvec = _get_sbert().encode([query], convert_to_tensor=True, normalize_embeddings=True).cpu().numpy()
        _, ids = idx.search(qvec, k); candidates += [sents[i] for i in ids[0]]
    # Wikipedia API with bias-aware queries
    if method in ("wiki", "all"):
        wiki_hits = []
        for phrase in key_phrases:
            for suffix in ["", " 편견", " 고정관념"]:
                q_term = phrase + suffix
                print(f"[Wiki] searching for bias context: '{q_term}'")
                q = quote(q_term)
                url = f"https://ko.wikipedia.org/w/api.php?action=query&list=search&srsearch={q}&format=json"
                res = requests.get(url, timeout=2).json()
                hits = res.get("query", {}).get("search", [])[:k]
                print(f"[Wiki] '{q_term}' -> {len(hits)} hits")
                for entry in hits:
                    title = entry.get("title", "")
                    print(f"[Wiki]   title: {title}")
                    sq = quote(title)
                    sum_url = f"https://ko.wikipedia.org/api/rest_v1/page/summary/{sq}"
                    r2 = requests.get(sum_url, timeout=2)
                    if r2.status_code == 200:
                        text = r2.json().get("extract", "").split(". ")[0] + "."
                        print(f"[Wiki]     summary: {text}")
                        wiki_hits.append(text)
        candidates += wiki_hits
    # Wiki dense
    if method in ("wiki_sbert", "all"):
        for kp in key_phrases:
            print(f"[WikiDense] embedding search for: '{kp}'")
            wiki_dense = get_wiki_by_embedding(kp, k)
            for sent in wiki_dense:
                print(f"[WikiDense] result: {sent}")
            candidates += wiki_dense
    # Dedupe + re-rank + MMR
    unique = list(dict.fromkeys(candidates))
    reranked = rerank_with_cross_encoder(query, unique)
    return mmr(query, reranked, k, mmr_lambda)