# CLAUDE.md

This file provides guidance to Claude Opus (claude.ai/code) when working with code in this repository.

## Project Overview

Vietnamese NLP Pipeline — self-contained system for text summarization, keyword extraction, and document topic clustering. All dependencies and models are bundled inside the repo.

## Commands

```bash
# Python environment
source venv/bin/activate
pip install -r requirements.txt

# Gradio web app (all-in-one, port 7860)
python app.py

# FastAPI backend (port 8000) — for React frontend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# React frontend (port 3000) — requires backend running
cd frontend && npm install && npm run dev
```

## Architecture Overview

The system has **3 entry points** and a **3-stage processing pipeline**:

```
Input document(s)
        │
        ▼
Stage 1: TextRank (backend/cluster/textrank/)
        Extract top sentences via VnCoreNLP word segmentation
        │
        ▼
Stage 2: KeyBERT-Vi (backend/cluster/keybert/)
        Extract keywords via PhoBERT + NER-ELECTRA
        │
        ▼
Stage 3: Gemini Clustering (backend/cluster/gemini_service.py)
        Classify into topic clusters (optional: GEMINI_API_KEY in .env)
        │
        ▼
Output: summary_sentences[], keywords[], topic_label[]
```

### Entry Points

| File | Type | Port | Purpose |
|------|------|------|---------|
| `app.py` | Gradio | 7860 | Web UI — all-in-one, no frontend needed |
| `backend/main.py` + `frontend/` | FastAPI + React | 8000 + 3000 | API backend + React SPA |
| `backend/cluster/combined_pipeline.py` | Python class | N/A | Programmatic / library usage |

## File Inventory

### Backend (Python)

```
backend/
  main.py                          — FastAPI server, 4 endpoints
  cluster/
    combined_pipeline.py          — Core pipeline: orchestrates all 3 stages
    gemini_service.py             — Gemini API wrapper for clustering
    textrank/
      textrank_facade.py          — TextRank facade (entry point for Stage 1)
      tools/
        parser.py                 — Sentence split + VnCoreNLP word-segment
        graph.py                  — Co-occurrence graph builder
        score.py                  — TextRank scoring (sum of connections, min-max norm)
        summarize.py              — Sentence selection for summary
        text.py                   — Data container (word_matrix, sentences, marks)
      stopwords/
        vietnamese.py             — Stopword loader
        Vietnamese-stopwords.txt  — 1940+ Vietnamese stopwords (underscore-joined)
    keybert/
      pipeline.py                 — KeyBERT-Vi pipeline (transformers.Pipeline subclass)
      vietnamese-stopwords-dash.txt — ~1998 stopwords for KeyBERT
      model/
        keyword_extraction_utils.py — All embedding / similarity / ngram functions
        named_entities.py         — NER via NER-ELECTRA
        process_text.py           — Text normalization (diacritics, punctuation)
```

### Frontend (React + TypeScript + Vite)

```
frontend/
  src/
    App.tsx                       — Main app, orchestrates upload → results flow
    main.tsx                      — React DOM entry
    services/api.ts               — FastAPI client (axios)
    types/index.ts                — TypeScript interfaces
    components/
      FileUpload.tsx              — Drag-and-drop file upload
      ResultsView.tsx             — Cluster sidebar + document cards + search/filter
```

### Pretrained Models (Git LFS)

```
pretrained-models/
  phobert.pt                      — PhoBERT full model (loaded via torch.load)
  ner-vietnamese-electra-base.pt — NER-ELECTRA full model
  vncorenlp/
    VnCoreNLP-1.2.jar             — VnCoreNLP JAR (word segmentation + POS tagging)
    models/
      wordsegmenter/              — VnCoreNLP word segmentation models
      postagger/                  — VnCoreNLP POS tagging models
      ner/                        — VnCoreNLP NER models
```

## Stage 1: TextRank Summarization

**File:** `backend/cluster/textrank/textrank_facade.py`
**Entry:** `TextRankFacade.summarize(text, max_sentences=None)`

### Data Flow

```
raw_text
   │
   ▼
Parser._get_sentences()       — regex split on [.? !]+ whitespace
   │
Parser._get_words(sentence)    — VnCoreNLP.word_segment() → underscore-joined
   │
Parser._filter_words()        — remove stopwords, punctuation, len<3
   │
Text (data container)          — word_matrix: {sentence_idx: {word_idx: word}}
   │                           sentences: {sentence_idx: original_sentence}
   │                           marks: sentence_idx → trailing punctuation
   ▼
Graph.create_graph()          — co-occurrence: each word connects to
   │                           prev/next word in same sentence
   │                           Output: {word: {sentence_idx: {word_idx: [connected_idx]}}}
   ▼
Score.calculate()            — for each word: sum of its connection counts
   │                           then min-max normalize to [0, 1]
   ▼
Summarize.get_summarize()     — top-N keywords weight sentences
   │                           top-K selected, re-sorted by original position
   ▼
List[str]                     — ordered summary sentences
```

### Key Classes

| Class | File | Role |
|-------|------|------|
| `Text` | `textrank/tools/text.py` | Data container: `word_matrix`, `sentences`, `marks` |
| `Parser` | `textrank/tools/parser.py` | Sentence split (regex `r'(?<=[.!?])\s+'`), VnCoreNLP wseg, stopword filter |
| `Graph` | `textrank/tools/graph.py` | Co-occurrence graph — each word links to immediate predecessor + successor |
| `Score` | `textrank/tools/score.py` | Sum of connection counts → min-max normalize to [0, 1] |
| `Summarize` | `textrank/tools/summarize.py` | Top-N keyword weights → top-K sentence selection, original order preserved |

### Auto-sentence-limit Logic

- `max_sentences` provided → use it (clamped to sentence count)
- `max_sentences=None`:
  - ≤ 5 sentences total → `min(3, n)`
  - > 5 sentences → `max(5, ceil(n * 0.4))`
- Keyword limit: `min(max(5, n), 15)`

## Stage 2: KeyBERT-Vi Keyword Extraction

**File:** `backend/cluster/keybert/pipeline.py`
**Entry:** `KeywordExtractorPipeline` (subclass of `transformers.Pipeline`)

### Data Flow

```
summary_text (output from Stage 1)
   │
process_text_pipeline()           — normalize diacritics, fix stuck sentences, strip
   │
KeywordExtractorPipeline.preprocess()
   │
KeywordExtractorPipeline._forward()
   │
   ├── get_segmentised_doc()               — VnCoreNLP word segment + NER-ELECTRA
   │                                        Entities merged with underscore
   │                                        e.g., "Hà Nội" → "Hà_Nội"
   ├── compute_filtered_text()              — VnCoreNLP POS tagging, keep only N/Np/V/Nc
   ├── get_doc_embeddings()                 — PhoBERT pooler_output
   │                                        sentence mean, first sentence weighted 2x
   ├── generate_ngram_list()
   │   ├── get_candidate_ngrams()           — ngrams in [ngram_low, ngram_high]
   │   │                                       filtered to N/Np/V/Nc, no stopwords, no numbers
   │   ├── add named entity ngrams           — PER/LOC/ORG from NER-ELECTRA
   │   ├── remove_overlapping_ngrams()       — remove shorter substrings
   │   └── limit_minimum_frequency()         — filter ngrams appearing < min_freq
   └── compute_ngram_embeddings()           — PhoBERT pooler_output per ngram
   │
KeywordExtractorPipeline.postprocess()
   │
   ├── compute_ngram_similarity()            — cosine similarity ngram vs doc embedding
   ├── remove_duplicates()                   — case-insensitive dedup (average scores)
   └── diversify_result_kmeans()             — K-means (100 iter) for topic diversity
   │
List[(keyword, score)]                — sorted by cosine similarity descending
```

### Key Functions (`keyword_extraction_utils.py`)

| Function | Purpose |
|----------|---------|
| `process_text_pipeline()` | Normalize: diacritics, special chars, stuck sentences, strip |
| `get_segmentised_doc()` | VnCoreNLP wseg + NER entity detection (underscore-merging) |
| `compute_filtered_text()` | VnCoreNLP POS tagging, keep N/Np/V/Nc tokens only |
| `get_doc_embeddings()` | PhoBERT pooler_output averaged over sentences; first sentence 2x |
| `compute_ngram_list()` | Generate n-grams from sub-sentences; filter stopwords, numbers |
| `get_candidate_ngrams()` | Intersection of actual ngrams and POS-filtered ngrams |
| `remove_overlapping_ngrams()` | Remove ngrams that are substrings of longer ngrams |
| `limit_minimum_frequency()` | Filter ngrams appearing < `min_freq` times |
| `compute_ngram_embeddings()` | PhoBERT embedding per ngram (pooler_output) |
| `compute_ngram_similarity()` | Cosine similarity between ngram and document embedding |
| `remove_duplicates()` | Average scores for case-insensitive duplicate ngrams |
| `diversify_result_kmeans()` | K-means clustering to pick diverse top-N keywords |

### NER (`named_entities.py`)

- Uses NER-ELECTRA via `NlpHUST/ner-vietnamese-electra-base` HuggingFace pipeline
- Sentence tokenization via `underthesea.sent_tokenize`
- Merges consecutive tokens with same entity tag into phrases
- Filters to entity types starting with "I" (inside tag), removes substrings
- PhoBERT tokenizer: `vinai/phobert-base-v2`

## Stage 3: Gemini Clustering

**File:** `backend/cluster/gemini_service.py`

Requires `GEMINI_API_KEY` in `.env`. Stage 3 is skipped entirely if the key is absent.

### 3-Step Workflow

```
New documents
    │
Step 1: extract_info_from_text()
        Gemini extracts 5-10 keyphrases + 1-sentence summary
        (keyphrases[:5] used for all subsequent steps — NEVER raw text)
    │
Step 2: assign_to_existing_clusters(new_docs, existing_clusters)
        Compare each new doc against existing cluster labels
        BE STRICT: assigns NEW_CLUSTER if no good match
        Can suggest renaming cluster labels
        Returns: {assignments, unassigned, renames: {old_label → new_label}}
    │
Step 3: cluster_unassigned_documents(docs)
        Gemini creates new coarse-grained Vietnamese labels
        Minimizes number of clusters
        No-match documents → "Linh tinh" cluster
    │
Output: List[DocumentCluster]
```

### Key Data Classes

```python
# backend/cluster/combined_pipeline.py
@dataclass
class AnalyzedDocument:
    id: str
    file_name: str = ""
    file_size: int = 0
    type: str = "text/plain"
    keyphrases: List[str] = field(default_factory=list)
    summary: str = ""

@dataclass
class DocumentCluster:
    label: str
    documents: List[AnalyzedDocument] = field(default_factory=list)

@dataclass
class TopicLabel:
    id: str
    name: str
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    document_count: int = 0

@dataclass
class ClassifyResult:
    assigned_label_ids: List[str] = field(default_factory=list)
    new_labels: List[TopicLabel] = field(default_factory=list)
    used_keywords: List[str] = field(default_factory=list)

@dataclass
class CombinedResult:
    original_text: str
    summary_sentences: List[str] = field(default_factory=list)
    summary_text: str = ""
    keywords: List[Tuple[str, float]] = field(default_factory=list)
    label_ids: List[str] = field(default_factory=list)
    title: Optional[str] = None
```

### Gemini API Settings

- Default model: `gemini-2.5-flash`
- `response_mime_type = "application/json"` with structured schemas
- Temperature: 0.1 (strict classification) to 0.3 (label generation)

## CombinedPipeline (`backend/cluster/combined_pipeline.py`)

**Class:** `CombinedPipeline`

```python
class CombinedPipeline:
    def __init__(
        self,
        top_n: int = 10,                    # number of keywords to extract
        ngram_n: Tuple[int, int] = (1, 3),   # ngram range [low, high]
        min_freq: int = 1,                   # minimum ngram frequency
        diversify_result: bool = False,      # K-means keyword diversification
        enable_clustering: bool = False,    # Gemini clustering (requires API key)
        gemini_api_key: Optional[str] = None,
    )
```

| Method | Signature | Purpose |
|--------|-----------|---------|
| `load()` | `() -> CombinedPipeline` | Lazy-load all models. Called on startup or first `run()` |
| `run()` | `(text, title=None, max_sentences=None) -> CombinedResult` | Single-doc pipeline: summarize → keywords → classify |
| `run_batch()` | `(texts, titles=None, max_sentences=None) -> (List[CombinedResult], List[TopicLabel])` | Multi-doc: extract all, then cluster all in one Gemini call |
| `reset_labels()` | `() -> None` | Clear all topic labels |
| `labels` | `property` | Get current list of `TopicLabel` |

### Model Loading Sequence (`load()`)

1. VnCoreNLP (`wseg + pos`) — single shared instance across all stages
2. PhoBERT from local `pretrained-models/phobert.pt` (`torch.load`, CPU)
3. NER-ELECTRA from local `pretrained-models/ner-vietnamese-electra-base.pt` (`torch.load`, CPU)
4. `TextRankFacade` + `KeywordExtractorPipeline` (both share the VnCoreNLP instance)
5. `GeminiService` (if `enable_clustering=True`)

## FastAPI Backend (`backend/main.py`)

**Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Health check: `{status, message, pipeline, gemini}` |
| `GET` | `/state` | Get current clusters and documents |
| `POST` | `/reset` | Clear all clusters and documents |
| `POST` | `/process-and-cluster` | Main pipeline: extract + cluster documents |

### `POST /process-and-cluster` Flow

1. Receive `ExtractRequest(texts: List[str], file_names: Optional[List[str]])`
2. For each text: call `pipeline.run()` → keywords + summary
3. Wrap in `AnalyzedDocument` objects
4. If existing clusters: call `gemini.assign_to_existing_clusters()` (Step 2)
5. For unassigned docs: call `gemini.cluster_unassigned_documents()` (Step 3)
6. Merge: existing + renamed + newly-assigned + new clusters
7. Return `{final_clusters, all_documents}`

### Lifespan

- Startup: load `CombinedPipeline(enable_clustering=True)` and `GeminiService`
- Shutdown: print "Server shutting down..."
- State is **in-memory** — restarting the server resets all clusters and documents

## React Frontend (`frontend/`)

### API Service (`services/api.ts`)

- `GET /state` — fetch current clusters and documents
- `POST /reset` — clear state
- `POST /process-and-cluster` — upload texts and file names

### App Flow

1. On mount: `GET /state` to sync with backend
2. User uploads files (PDF or TXT) via drag-and-drop
3. Files read client-side via `FileReader.readAsText()`
4. Call `/process-and-cluster` with texts + file names
5. Display: cluster sidebar (left) + document card grid (right) + search/filter bar
6. Each card shows: file icon, filename, summary excerpt, top-5 keyphrases

### TypeScript Interfaces (`types/index.ts`)

```typescript
interface AnalyzedDocument {
    id: string;
    file_name: string;
    file_size: number;
    type: string;
    keyphrases: string[];
    summary: string;
}

interface DocumentCluster {
    label: string;
    documents: AnalyzedDocument[];
}

interface AnalyzeResult {
    keywords: Array<[string, number]>;
    summary: string;
    summary_sentences: string[];
    label_ids: string[];
}

interface ProcessResponse {
    final_clusters: DocumentCluster[];
    all_documents: AnalyzedDocument[];
}
```

## Gradio App (`app.py`)

3-tab interface:

| Tab | Function | Purpose |
|-----|----------|---------|
| Single Document | `process_single()` | Title + text input, configurable params, output: summary + keywords + stats |
| Multi-Document Batch | `process_batch()` | Multiple docs (separated by `===`), per-doc details + cluster overview |
| Label Space | `process_label_space()` | Full 3-step Gemini workflow: extract → assign → cluster |

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Shared VnCoreNLP instance** | Both TextRank and KeyBERT reuse one VnCoreNLP instance (`wseg + pos`), saving RAM (~200MB) |
| **Lazy loading** | Models load only when `load()` is called or first `run()`. Critical for large models |
| **Keyphrases-only to Gemini** | Raw text is NEVER sent to Gemini. Only `keyphrases[:5]` are used, keeping API calls fast and cheap |
| **Transformers constraint** | `transformers>=4.30,<4.40` required — NER-ELECTRA uses `ElectraAttention` incompatible with transformers 5.x |
| **PhoBERT + NER from local `.pt`** | Models stored in `pretrained-models/` and loaded via `torch.load` (not `from_pretrained`), avoids HuggingFace cache issues |
| **In-memory clustering state** | `_clusters`, `_all_documents` are FastAPI module-level variables. No persistence — restart resets all state |
| **Git LFS for models** | `phobert.pt`, `ner-vietnamese-electra-base.pt`, `vncorenlp/` stored in Git LFS to avoid bloating the repo |
| **PhoBERT tokenizer from hub** | While the model `.pt` is local, the tokenizer (`vinai/phobert-base-v2`) is loaded from HuggingFace Hub |
| **No server-side PDF parsing** | React frontend reads files client-side via `FileReader`. Backend expects plain text only |