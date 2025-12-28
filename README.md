# TED Talk RAG (Course Submission)

Built a retrieval-augmented QA system over the TED dataset using provided models and infrastructure. The pipeline ingests transcripts, chunks and embeds them into Pinecone, then answers user queries by retrieving relevant context and feeding it to a language model.

## Core Components

**Data & Models**: Used `data/ted_talks_en.csv`, embedding model RPRTHPB-text-embedding-3-small, and chat model RPRTHPB-gpt-5-mini via provided endpoints.

**Implementation**: Batched, resumable ingest with checkpointing. RAGService handles embed → retrieve → prompt → chat. Two endpoints: `/api/prompt` (answers) and `/api/stats` (config).

**Hyperparameter tuning**: See the structured section below for the two-step process and final selection.

## Chunking Strategy
Token-based with configurable overlap. Each chunk stored with talk metadata: talk_id, title, speaker, topics, description, recorded_date, url, chunk_id, text.

## Ingestion
Script: [ingest.py](ingest.py)

Required env: `PINECONE_API_KEY`, `PINECONE_HOST`, `MODEL_BASE_URL`, `MODELS_API_KEY`.

Features: Checkpointing via `ingest_progress.txt`, batching (default 32), checkpoint-every (default 10 talks).

```bash
python ingest.py --dataset data/ted_talks_en.csv --batch-size 32 --checkpoint-every 10
```

## API Endpoints
- POST `/api/prompt`: body `{ "question": str }` → returns answer, context snippets, augmented prompt. The server enforces `top_k` from config (no client override).
- GET `/api/stats`: returns current `chunk_size`, `overlap_ratio`, `top_k` from config.

## Hyperparameter Tuning

Below is a concise, structured summary of how we tuned the system and what we ultimately chose.

**Step 1 — Subset Sweep (50 talks)**
- Dimensions: chunk_size ∈ {600, 1300, 2000}, overlap ∈ {0.05, 0.15, 0.25}, top_k ∈ {5, 7, 10, 15}.
- Metrics: compliance (4 checks aligned with the assignment’s four task examples) and mean similarity.
- Outcome: initial pick **chunk_size=600, overlap=0.15, top_k=5** — strongest compliance and similarity on the subset at reasonable cost.
- Raw results: [grid_summary.json](eval_results/size=600_overlap=0.15/grid_summary.json)

**Subset results — all runs (passed/4, mean_similarity, distinct_talks, prompt_chars)**

### size=600_overlap=0.05 ([grid_summary.json](eval_results/size=600_overlap=0.05/grid_summary.json))
| top_k | passed/4 | mean_avg_score | total_distinct | prompt_chars |
| --- | --- | --- | --- | --- |
| 5 | 4 | 0.4220 | 15 | 88488 |
| 7 | 3 | 0.4064 | 21 | 128762 |
| 10 | 4 | 0.3897 | 30 | 188890 |
| 15 | 4 | 0.3713 | 43 | 290945 |

### size=600_overlap=0.15 ([grid_summary.json](eval_results/size=600_overlap=0.15/grid_summary.json))
| top_k | passed/4 | mean_avg_score | total_distinct | prompt_chars |
| --- | --- | --- | --- | --- |
| 5 | 4 | 0.4384 | 14 | 47638 |
| 7 | 4 | 0.4270 | 16 | 65020 |
| 10 | 4 | 0.4121 | 20 | 92426 |
| 15 | 4 | 0.3954 | 30 | 141596 |

### size=600_overlap=0.25 ([grid_summary.json](eval_results/size=600_overlap=0.25/grid_summary.json))
| top_k | passed/4 | mean_avg_score | total_distinct | prompt_chars |
| --- | --- | --- | --- | --- |
| 5 | 3 | 0.4393 | 13 | 49019 |
| 7 | 3 | 0.4304 | 16 | 66141 |
| 10 | 4 | 0.4183 | 21 | 93133 |
| 15 | 4 | 0.4010 | 31 | 142250 |

### size=1300_overlap=0.05 ([grid_summary.json](eval_results/size=1300_overlap=0.05/grid_summary.json))
| top_k | passed/4 | mean_avg_score | total_distinct | prompt_chars |
| --- | --- | --- | --- | --- |
| 5 | 4 | 0.4220 | 15 | 88488 |
| 7 | 4 | 0.4064 | 21 | 128762 |
| 10 | 4 | 0.3897 | 30 | 188890 |
| 15 | 4 | 0.3713 | 43 | 290945 |

### size=1300_overlap=0.15 ([grid_summary.json](eval_results/size=1300_overlap=0.15/grid_summary.json))
| top_k | passed/4 | mean_avg_score | total_distinct | prompt_chars |
| --- | --- | --- | --- | --- |
| 5 | 4 | 0.4189 | 14 | 84184 |
| 7 | 4 | 0.4041 | 19 | 124027 |
| 10 | 4 | 0.3853 | 26 | 174312 |
| 15 | 4 | 0.3684 | 37 | 267367 |

### size=1300_overlap=0.25 ([grid_summary.json](eval_results/size=1300_overlap=0.25/grid_summary.json))
| top_k | passed/4 | mean_avg_score | total_distinct | prompt_chars |
| --- | --- | --- | --- | --- |
| 5 | 4 | 0.4103 | 13 | 81590 |
| 7 | 4 | 0.4055 | 18 | 113854 |
| 10 | 4 | 0.3923 | 24 | 177652 |
| 15 | 4 | 0.3739 | 38 | 273122 |

### size=2000_overlap=0.05 ([grid_summary.json](eval_results/size=2000_overlap=0.05/grid_summary.json))
| top_k | passed/4 | mean_avg_score | total_distinct | prompt_chars |
| --- | --- | --- | --- | --- |
| 5 | 4 | 0.4194 | 17 | 108520 |
| 7 | 3 | 0.4009 | 23 | 145530 |
| 10 | 3 | 0.3806 | 32 | 232596 |
| 15 | 3 | 0.3605 | 47 | 365144 |

### size=2000_overlap=0.15 ([grid_summary.json](eval_results/size=2000_overlap=0.15/grid_summary.json))
| top_k | passed/4 | mean_avg_score | total_distinct | prompt_chars |
| --- | --- | --- | --- | --- |
| 5 | 4 | 0.4237 | 14 | 120625 |
| 7 | 3 | 0.4052 | 22 | 163570 |
| 10 | 4 | 0.3862 | 32 | 233708 |
| 15 | 3 | 0.3657 | 43 | 381694 |

### size=2000_overlap=0.25 ([grid_summary.json](eval_results/size=2000_overlap=0.25/grid_summary.json))
| top_k | passed/4 | mean_avg_score | total_distinct | prompt_chars |
| --- | --- | --- | --- | --- |
| 5 | 4 | 0.4181 | 15 | 110755 |
| 7 | 3 | 0.4043 | 19 | 165971 |
| 10 | 3 | 0.3877 | 27 | 237375 |
| 15 | 4 | 0.3672 | 38 | 390307 |

**Step 2 — Full-Index top_k Sweep**
- Fixed chunking at the Step 1 choice (size=600, overlap=0.15).
- Searched k ∈ {5, 10, 15, 20, 25, 30}; settled on **top_k=20** for improved distinct-talk diversity while keeping prompt size moderate.

### Final k Sweep (full index; size=600, overlap=0.15)
| top_k | passed/4 | mean_avg_score | total_distinct | prompt_chars |
| --- | --- | --- | --- | --- |
| 5 | 3 | 0.5277 | 16 | 50065 |
| 10 | 3 | 0.5112 | 30 | 96944 |
| 15 | 3 | 0.4972 | 31 | 106682 |
| 20 | 3 | 0.4932 | 54 | 189274 |
| 25 | 4 | 0.4872 | 66 | 236244 |
| 30 | 3 | 0.4824 | 78 | 281750 |

Source JSON: [grid_summary.json](eval_results_full_sweep/size=600_overlap=0.15/grid_summary.json)

**Final Selection**
- Parameters: **chunk_size=600**, **overlap=0.15**, **top_k=20**.
- Why this model: balances diversity and brevity with moderate prompt sizes.
- Reflected in code: see [config.py](config.py) and the `/api/stats` endpoint.




## Reproduction
1. Set env vars: `PINECONE_API_KEY`, `PINECONE_HOST`, `MODEL_BASE_URL`, `MODELS_API_KEY`.
2. Ingest: `python ingest.py` (checkpoints to progress file; safe to rerun).
3. Deploy or run locally; POST to `/api/prompt` with `{"question": "..."}`.
4. To re-eval: `python tools/eval_grid.py --base-url <url>` then `python tools/score_runs.py`.

## Code Map
- [ingest.py](ingest.py): Chunk, embed, upsert with batching and checkpoints.
- [rag.py](rag.py): RAGService (embed query, retrieve, build prompt, chat).
- [api/prompt.py](api/prompt.py): `/api/prompt` endpoint.
- [api/stats.py](api/stats.py): `/api/stats` endpoint.
- [config.py](config.py): Configuration defaults.
- [tools/](tools): Evaluation scripts.