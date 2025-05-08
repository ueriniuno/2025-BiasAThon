# retriever.py
import os, pickle, faiss, threading
from sentence_transformers import SentenceTransformer

_MODEL_NAME  = "jhgan/ko-sbert-nli"          # 한국어 SBERT
_DB_TXT      = "bias_db.txt"                # 240 문장 파일
_IDX_FILE    = "faiss_bias.index"
_SENT_FILE   = "bias_sent.pkl"

# ---- (NEW) lazy-singleton  -------------------------
_SBERT_MODEL = None          # 전역 캐시
_SBERT_LOCK  = threading.Lock()

def _get_sbert():
    """SBERT를 단 1번만 메모리에 올림 + warm-up"""
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        with _SBERT_LOCK:
            if _SBERT_MODEL is None:
                print("🔧  SBERT 로딩…")
                _SBERT_MODEL = SentenceTransformer(_MODEL_NAME, device="mps")
                _SBERT_MODEL.encode(["warm-up"], convert_to_tensor=True)
                print("✅  SBERT 로딩 완료")
    return _SBERT_MODEL
# ----------------------------------------------------


def build_index(db_txt: str = _DB_TXT):
    """bias_db.txt → 임베딩 → FAISS 인덱스 (*.index / *.pkl) 저장"""
    sents = [ln.strip() for ln in open(db_txt, encoding="utf-8") if ln.strip()]
    embs  = _get_sbert().encode(
        sents, convert_to_tensor=True, normalize_embeddings=True
    ).cpu().numpy()

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, _IDX_FILE)
    pickle.dump(sents, open(_SENT_FILE, "wb"))
    print(f"✅  Bias DB indexed: {len(sents)} 문장")

def get_relevant(query: str, k: int = 2):
    """query → Top-k 문장 리스트"""
    if not os.path.exists(_IDX_FILE):
        build_index()
    index  = faiss.read_index(_IDX_FILE)
    sents  = pickle.load(open(_SENT_FILE, "rb"))
    qvec   = _get_sbert().encode(
        [query], convert_to_tensor=True, normalize_embeddings=True
    ).cpu().numpy()
    _, idx = index.search(qvec, k)
    return [sents[i] for i in idx[0]]