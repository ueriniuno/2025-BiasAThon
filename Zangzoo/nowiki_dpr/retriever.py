import os
import pickle
import faiss
import threading
from urllib.parse import quote
import numpy as np
import torch

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
from keybert import KeyBERT
from transformers import (
    DPRQuestionEncoder,
    DPRContextEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoderTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer
)

# 환경 변수로 토글
DB_TXT        = os.getenv("DB_TXT", "bias_db.txt")
IDX_FILE      = os.getenv("IDX_FILE", "faiss_bias.index")
SENT_FILE     = os.getenv("SENT_FILE", "bias_sent.pkl")
WIKI_IDX      = os.getenv("WIKI_IDX", "faiss_wiki.index")
WIKI_SENTS    = os.getenv("WIKI_SENTS", "wiki_sent.pkl")
MMR_K         = int(os.getenv("MMR_K", "2"))
MMR_LAMBDA    = float(os.getenv("MMR_LAMBDA", "0.8"))
RETRIEVAL_K   = int(os.getenv("RETRIEVAL_K", "5"))

# KeyBERT
_kw = KeyBERT("jhgan/ko-sbert-nli")

# BM25 singleton
_BM25, _BM25_SENTS = None, None
def _init_bm25():
    global _BM25, _BM25_SENTS
    if _BM25 is None:
        sents = [ln.strip() for ln in open(DB_TXT, encoding="utf-8") if ln.strip()]
        tokenized = [s.split() for s in sents]
        _BM25 = BM25Okapi(tokenized)
        _BM25_SENTS = sents
    return _BM25, _BM25_SENTS

# SBERT singleton
_SBERTER, _SBERT_LOCK = None, threading.Lock()
def _get_sbert():
    global _SBERTER
    if _SBERTER is None:
        with _SBERT_LOCK:
            if _SBERTER is None:
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                _SBERTER = SentenceTransformer("jhgan/ko-sbert-nli", device=device)
                _SBERTER.encode(["warm-up"], convert_to_tensor=True)
    return _SBERTER

# CrossEncoder ensemble (no tqdm)
os.environ["SENTENCE_TRANSFORMERS_NO_TQDM"] = "1"
_CE_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/stsb-distilroberta-base",
    "monologg/koelectra-base-v3-discriminator"
]
_CE_ENSEMBLE = [
    CrossEncoder(name)
    for name in _CE_MODELS
]

# DPR encoders
device = "mps" if torch.backends.mps.is_available() else "cpu"
_q_enc = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
_c_enc = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
_q_tok = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
_c_tok = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

def encode_question_dpr(q: str):
    inp = _q_tok(q, return_tensors="pt", truncation=True, padding=True).to(device)
    return _q_enc(**inp).pooler_output  # [1,dim]

def encode_ctx_dpr(doc: str):
    inp = _c_tok(doc, return_tensors="pt", truncation=True, padding=True).to(device)
    return _c_enc(**inp).pooler_output

# Query expansion T5
_t5 = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
_t5_tok = T5Tokenizer.from_pretrained("t5-small")
def expand_query(q: str) -> str:
    inp = _t5_tok(f"paraphrase: {q}", return_tensors="pt", truncation=True).to(device)
    out = _t5.generate(**inp, max_length=64)
    return _t5_tok.decode(out[0], skip_special_tokens=True)

# CrossEncoder re-rank
def rerank_with_cross_encoder(query: str, docs: list[str]) -> list[str]:
    pairs = [[query, d] for d in docs]
    scores = [ce.predict(pairs) for ce in _CE_ENSEMBLE]
    avg = np.mean(scores, axis=0)
    return [doc for _, doc in sorted(zip(avg, docs), key=lambda x: x[0], reverse=True)]

# MMR
def mmr(query: str, docs: list[str], k: int, lambda_param: float) -> list[str]:
    sbert = _get_sbert()
    q_emb = sbert.encode([query], convert_to_tensor=True, normalize_embeddings=True)
    d_embs = sbert.encode(docs, convert_to_tensor=True, normalize_embeddings=True)
    rel = cos_sim(q_emb, d_embs)[0].cpu().tolist()
    selected, idxs, cand = [], [], list(range(len(docs)))
    for _ in range(min(k, len(cand))):
        if not selected:
            i = int(np.argmax(rel))
        else:
            scores = []
            for j in cand:
                div = max(cos_sim(d_embs[j].unsqueeze(0), d_embs[idxs]).cpu().tolist()[0])
                scores.append((j, lambda_param * rel[j] - (1-lambda_param) * div))
            i = max(scores, key=lambda x: x[1])[0]
        selected.append(docs[i]); idxs.append(i); cand.remove(i)
    return selected

# --- Build DB FAISS index ---
def build_db_index(db_txt: str = DB_TXT):
    sents = [ln.strip() for ln in open(db_txt, encoding="utf-8") if ln.strip()]
    embs = _get_sbert().encode(sents, convert_to_tensor=True, normalize_embeddings=True).cpu().numpy()
    idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs)
    faiss.write_index(idx, IDX_FILE)
    pickle.dump(sents, open(SENT_FILE, "wb"))
    print(f"✅  DB indexed: {len(sents)} sentences")
    
# Hybrid retrieve: BM25 → DPR → CE → MMR
def hybrid_retrieve(query: str,
                    k_bm25: int = 100,
                    k_dpr:   int = 20,
                    k_ce:    int = 5,
                    final_k: int = 2) -> list[str]:
    # 1) BM25로 빠르게 후보 뽑기
    bm25, sents = _init_bm25()
    tokens = query.split()
    bm25_scores = bm25.get_scores(tokens)
    # scores를 내림차순 정렬한 인덱스에서 상위 k_bm25개 추출
    top_bm25 = [sents[i] for i in np.argsort(bm25_scores)[-k_bm25:]][::-1]

    # 2) DPR로 후보 재정렬
    q_vec  = encode_question_dpr(query)                     # [1, dim]
    d_vecs = torch.cat([encode_ctx_dpr(d) for d in top_bm25], dim=0)  # [k_bm25, dim]
    sims   = (q_vec @ d_vecs.T).squeeze().cpu().tolist()    # [k_bm25]
    # 유사도 높은 순으로 인덱스 정렬 후 상위 k_dpr개 문장 선택
    top_dpr = [top_bm25[i] for i in np.argsort(sims)[-k_dpr:]][::-1]

    # 3) Cross-Encoder 재정렬
    ce_reranked = rerank_with_cross_encoder(query, top_dpr)[:k_ce]

    # 4) MMR로 다양성 고려해 최종 선택
    return mmr(query, ce_reranked, k=final_k, lambda_param=MMR_LAMBDA)

# Unified interface
def get_relevant(query: str, method: str="hybrid", **kwargs) -> list[str]:
    if kwargs.get("use_expansion"):
        query = expand_query(query)
    if method == "hybrid":
        return hybrid_retrieve(
            query,
            k_bm25=kwargs.get("k_bm25", 100),
            k_dpr=  kwargs.get("k_dpr",   20),
            k_ce=   kwargs.get("k_ce",    5),
            final_k=kwargs.get("final_k", 2)
        )
    raise ValueError("method must be 'hybrid'")

