# Reading Guide: `mfa align` Command

## 1. CLI Entry Point

Start here to understand how the command is invoked:

- **`command_line/mfa.py`** — the `mfa` CLI group (Click router)
- **`command_line/align.py`** — `align_corpus_cli()` at line 107; parses all CLI args, instantiates `PretrainedAligner`, calls `.align()`

---

## 2. Top-Level Orchestrator

- **`alignment/pretrained.py`** — `PretrainedAligner` class:
  - `__init__()` (line 78) — opens the `.zip` acoustic model, extracts params
  - `parse_parameters()` (line 304) — merges YAML config with CLI args
  - `setup()` (line 234) — initializes DB, loads model files, loads corpus
  - `align()` — calls into `CorpusAligner.align()`

---

## 3. Inheritance Chain (read in order)

```
PretrainedAligner
  └── CorpusAligner         → alignment/base.py        (core align logic)
        └── AlignMixin      → alignment/mixins.py       (config params)
        └── AcousticCorpusPronunciationMixin → corpus loading
  └── TopLevelMfaWorker     → abc.py                   (lifecycle mgmt)
        └── DatabaseMixin   → abc.py                   (SQLite/PostgreSQL)
```

Read **`alignment/mixins.py`** and **`alignment/base.py`** together — they define the alignment configuration and the actual alignment steps.

---

## 4. Core Alignment Steps (in `alignment/base.py`)

| Method | Line | What it does |
|---|---|---|
| `compile_train_graphs()` | ~370 | Builds HCLG FSTs (phone→lexicon→grammar) |
| `_align()` | ~404 | Runs Kaldi's GmmAligner in parallel |
| `collect_alignments()` | ~902 | Reads Kaldi output, writes to DB |
| `analyze_alignments()` | ~178 | Computes SNR, duration Z-scores |
| `export_files()` | ~1395 | Writes TextGrid/JSON/CSV output |

---

## 5. Parallel Workers

**`alignment/multiprocessing.py`** — the actual computation happens here, in separate processes:

| Class | Role |
|---|---|
| `CompileTrainGraphsFunction` | FST compilation per job |
| `AlignFunction` | Kaldi alignment per job |
| `AnalyzeAlignmentsFunction` | Stats per utterance |
| `ExportTextGridProcessWorker` | File writing per utterance |
| `FineTuneFunction` | Millisecond-precision refinement (optional) |

---

## 6. Supporting Files (read as needed)

| File | When to read it |
|---|---|
| `db.py` | To understand the ORM schema (`Utterance`, `PhoneInterval`, `WordInterval`, etc.) |
| `models.py` | To understand how the `.zip` acoustic model is loaded |
| `dictionary/mixins.py` | To understand pronunciation dictionary handling |
| `corpus/acoustic_corpus.py` | To understand how files/speakers/utterances are scanned |
| `data.py` | Enums like `WorkflowType`, `PhoneType` referenced throughout |
| `abc.py` | Base classes for database lifecycle and worker management |

---

## 7. Full Data Flow

```
CLI Arguments
    ↓
align_corpus_cli()
    ↓
PretrainedAligner.__init__() + parse_parameters()
    ↓
PretrainedAligner.align()
    ↓
PretrainedAligner.setup()
    ├─ Database creation (SQLite/PostgreSQL)
    ├─ Acoustic model extraction & phone/grapheme table population
    ├─ Corpus loading (File, Speaker, Utterance, Word, Pronunciation entries)
    └─ KalpyAligner instantiation
    ↓
CorpusAligner.align()
    ├─ compile_train_graphs() → CompileTrainGraphsFunction (Kaldi FSTs)
    ├─ _align()              → AlignFunction (Kaldi GmmAligner, parallel jobs)
    └─ collect_alignments()  → PhoneInterval/WordInterval DB population
    ↓
analyze_alignments() → AnalyzeAlignmentsFunction (SNR, duration stats)
    ↓
export_files() → ExportTextGridProcessWorker (TextGrid/JSON/CSV output)
    ↓
[OPTIONAL] fine_tune_alignments() → FineTuneFunction (millisecond precision)
    ↓
[OPTIONAL] evaluate_alignments() (compare vs reference alignments)
    ↓
cleanup() (close DB, clean temp files)
```

---

## 8. Suggested Reading Order

1. `command_line/align.py` — see the full CLI interface
2. `alignment/pretrained.py` — the orchestrator
3. `alignment/mixins.py` — configuration knobs
4. `alignment/base.py` — the core pipeline
5. `alignment/multiprocessing.py` — the parallel workers
6. `db.py` — the data model
