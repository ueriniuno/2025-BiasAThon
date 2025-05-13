#retriever.py

import os,re, requests
import pickle
import faiss
import threading
from sentence_transformers import SentenceTransformer,CrossEncoder
from rank_bm25 import BM25Okapi
try:
    from konlpy.tag import Okt
    okt = Okt()
except:
    okt = None
import numpy as np
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# SBERT & DB 설정
_MODEL_NAME  = "jhgan/ko-sbert-nli"
_DB_TXT      = "bias_db.txt"
_IDX_FILE    = "faiss_bias.index"
_SENT_FILE   = "bias_sent.pkl"

# 앙상블용 Cross-Encoder 모델들 (영어+한국어 특화)
_CE_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/stsb-distilroberta-base",
    "monologg/koelectra-base-v3-discriminator"
]
_CE_ENSEMBLE = [CrossEncoder(m) for m in _CE_MODELS]

# --- BM25 초기화 ---
_BM25 = None
_BM25_SENTS = None
def _init_bm25(db_txt: str = _DB_TXT):
    global _BM25, _BM25_SENTS
    if _BM25 is None:
        sents = [ln.strip() for ln in open(db_txt, encoding="utf-8") if ln.strip()]
        if okt:
            tokenized = [okt.morphs(sent) for sent in sents]
        else:
            tokenized = [sent.split() for sent in sents]
        _BM25 = BM25Okapi(tokenized)
        _BM25_SENTS = sents
    return _BM25, _BM25_SENTS

# --- SBERT 모델 로더 (singleton) ---
_SBERT_MODEL = None
_SBERT_LOCK  = threading.Lock()
def _get_sbert():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        with _SBERT_LOCK:
            if _SBERT_MODEL is None:
                print("🔧  SBERT 로딩…")
                _SBERT_MODEL = SentenceTransformer(_MODEL_NAME, device="mps")
                _SBERT_MODEL.encode(["warm-up"], convert_to_tensor=True)
                print("✅  SBERT 로딩 완료")
    return _SBERT_MODEL

# --- Cross-Encoder 재랭킹 (앙상블) ---
def rerank_with_cross_encoder(query: str, docs: list[str]) -> list[str]:
    """
    Cross-Encoder 앙상블로 후보 리스트를 재랭킹합니다.
    각 모델로 예측한 점수를 평균 내림차순으로 정렬합니다.
    """
    pairs = [[query, d] for d in docs]
    all_scores = [ce.predict(pairs) for ce in _CE_ENSEMBLE]
    avg_scores = np.mean(all_scores, axis=0)
    # 점수 순서대로 도큐먼트 정렬
    ranked     = [doc for _, doc in sorted(zip(avg_scores, docs), key=lambda x: x[0], reverse=True)]
    return ranked

# --- MMR (Maximal Marginal Relevance) ---
def mmr(query: str, doc_sents: list[str], k: int, lambda_param: float) -> list[str]:
    sbert = _get_sbert()
    q_emb = sbert.encode([query], convert_to_tensor=True)
    d_embs = sbert.encode(doc_sents, convert_to_tensor=True)
    selected, selected_idx, candidates = [], [], list(range(len(doc_sents)))
    rel_scores = cos_sim(q_emb, d_embs)[0].cpu().tolist()
    for _ in range(min(k, len(candidates))):
        if not selected:
            idx = int(np.argmax(rel_scores))
        else:
            mmr_scores = []
            for i in candidates:
                diversity = max(cos_sim(d_embs[i].unsqueeze(0), d_embs[selected_idx]).cpu().tolist()[0])
                score = lambda_param * rel_scores[i] - (1 - lambda_param) * diversity
                mmr_scores.append((i, score))
            idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(doc_sents[idx])
        selected_idx.append(idx)
        candidates.remove(idx)
    return selected

# --- FAISS 인덱스 빌드 ---
def build_index(db_txt: str = _DB_TXT):
    sents = [ln.strip() for ln in open(db_txt, encoding="utf-8") if ln.strip()]
    embs  = _get_sbert().encode(
        sents, convert_to_tensor=True, normalize_embeddings=True
    ).cpu().numpy()
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, _IDX_FILE)
    pickle.dump(sents, open(_SENT_FILE, "wb"))
    print(f"✅  Bias DB indexed: {len(sents)} 문장")

# --- 통합 검색 인터페이스 ---
def get_relevant(query: str, k: int=5, method: str="all", pre_k: int=50, mmr_lambda: float=0.9) -> list[str]:
    """
    method:
      - 'bm25': BM25 상위 k
      - 'sbert': SBERT(FAISS) 상위 k
      - 'all':   BM25 + SBERT 조합
    """
    candidates = []
    # BM25
    if method in ("bm25", "all"):
        bm25, sents = _init_bm25()
        tokens      = okt.morphs(query) if okt else query.split()
        scores      = bm25.get_scores(tokens)
        top_idxs    = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        candidates.extend([sents[i] for i in top_idxs])
    # SBERT(FAISS)
    if method in ("sbert", "all"):
        if not os.path.exists(_IDX_FILE):
            build_index()
        index  = faiss.read_index(_IDX_FILE)
        sents  = pickle.load(open(_SENT_FILE, "rb"))
        qvec   = _get_sbert().encode([query], convert_to_tensor=True, normalize_embeddings=True).cpu().numpy()
        _, idxs= index.search(qvec, k)
        candidates.extend([sents[i] for i in idxs[0]])
    # 중복 제거
    return list(dict.fromkeys(candidates))