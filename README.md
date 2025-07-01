# 2025-BiasAThon
# 🧠 LLaMA-3.1-8B 편향 완화 프로젝트

RAG와 프롬프트 엔지니어링을 통해 **LLaMA-3.1-8B-Instruct 모델**의 사회적 편향을 완화하고, 공정한 응답을 생성하는 프로젝트입니다.

---

## ✨ 프로젝트 개요

- LLM은 훈련 데이터의 특성상 **사회적 편향**을 내포할 수 있으며, 이는 응답에 그대로 드러날 수 있습니다.
- 본 프로젝트에서는 **RAG (Retrieval-Augmented Generation)** 와 **프롬프트 엔지니어링**을 활용하여,
  LLaMA-3.1-8B-Instruct 모델이 **균형 잡힌 응답**을 생성하도록 유도합니다.
- 실험 대상은 사회적 편향이 내포된 다양한 질문들과, 그에 대한 응답의 변화입니다.

---

## 🔧 기술 스택

- 💬 **LLaMA-3.1-8B-Instruct** (Meta)
- 🔎 **RAG (FAISS + BM25)** 기반 문서 검색
- 🧩 **Prompt Engineering** (편향 유도/완화 방식 비교)
- 🧠 **MMR / Cross-Encoder reranking**
- 🗂️ Python, LangChain, Transformers, SentenceTransformers, Pinecone (or FAISS)

---

## 🗂️ 폴더 구조 안내

본 프로젝트는 각 팀원이 독립적인 구조로 실험을 진행했습니다.  
각자의 폴더 아래에 RAG 구조 및 추론 방식 실험 코드가 포함되어 있습니다.  
📦project-root/  
├── Yerin// # 모델 추론 방식 개선  
│ ├── 📂main.py # 전체 실행 스크립트 (retriever + inference + 저장)  
│ ├── 📂model_runner.py # LLaMA-3.1-8B 모델 실행 (pipeline + 양자화 + GPU 최적화)  
│ ├── 📂prompt_engineer.py # 프롬프트 구성 및 편향 완화 템플릿 생성  
│ ├── 📂postprocessor.py # 모델 응답 후처리 및 평가  
│ └── 📂data_loader.py # 테스트용 데이터셋 불러오기  
├── Zangzoo/ # RAG를 위한 위키 적용  
├── Yeogyeong/ # 프롬프트 기반 편향 완화 실험  
├── Sally/ # 초기 실험용 코드  
├── .idea/ # IDE 설정 파일 (무시 가능)  

---

## 👥 팀원 및 역할

| 이름       | 역할 |
|------------|-----------|
| 정예린      | RAG 문서 구축, 프롬프트 설계 |
| 장지우    | RAG 문서 구축 및 검색 전략 비교 실험 |
| 송여경  | 편향 완화 프롬프트 구조 실험, 프롬프트 템플릿 설계 |
| 이채연      | 데이터 구성 및 베이스라인 작성 |


