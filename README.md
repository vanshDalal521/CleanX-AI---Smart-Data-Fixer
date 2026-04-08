---
title: CleanX AI
emoji: 🧹
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
tags:
- openenv
- data-cleaning
pinned: false
---

# CleanX AI - Smart Data Fixer (OpenEnv)


CleanX AI is a high-fidelity environment designed to evaluate AI agents on real-world data cleaning tasks. It simulates scenarios where an agent must transform messy, inconsistent CSV data into a clean, structured schema.

---

## 🛠️ Environment Description
Data cleaning is a critical, time-consuming task for humans. This environment provides a safe, reproducible sandbox where agents can interact with dataframes using a set of strictly defined operations.

### Action Space
The agent can perform the following operations via a JSON interface:
- **`drop_row`**: Options for `dropna` or `drop_duplicates`.
- **`rename_column`**: Changes column descriptors.
- **`cast_type`**: Supports `datetime`, `float`, `bool`, `int`, and `str`.
- **`submit`**: Finalizes the task and triggers the grader.

### Observation Space
- **`dataset_preview`**: A CSV snippet representation of the current dataframe state.
- **`columns`**: List of current column names.
- **`shape`**: Current dimensions of the data.
- **`goal`**: A human-readable text goal for the task.
- **`progress`**: A real-time score from **0.0 to 1.0** reflecting task completion.

---

## 🎯 Tasks & Difficulty
| Task | Difficulty | Real-World Context |
|---|---|---|
| **Easy** | Easy | Contact list standardization (Renaming names, dropping nulls). |
| **Medium** | Medium | E-commerce logs (De-duplication, date casting, currency stripping). |
| **Hard** | Hard | Sensor logs (ERR string handling, multi-step casting, complex renaming). |

---

## 🚀 Setup & Usage

### 1. Local Development
*   **Install dependencies**:
    ```bash
    pip install -e .
    pip install -r server/requirements.txt
    ```
*   **Launch Server**:
    ```bash
    python -m server.app --port 8000
    ```
*   **Run Inference**:
    ```bash
    export HF_TOKEN="your_key"
    python inference.py
    ```

### 2. Docker
```bash
docker build -t cleanx-ai .
docker run -p 8000:8000 cleanx-ai
```

---

## 📊 Baseline Scores
Baseline scores obtained using **GPT-4o** with the provided `inference.py`:
*   **Easy**: 1.0
*   **Medium**: 1.0
*   **Hard**: 1.0

## 📝 OpenEnv Compliance
This project implements the full OpenEnv specification:
*   **Typed Models**: Pydantic models for Action/Observation.
*   **REST API**: Supports `reset()`, `step()`, and `state()` via FastAPI.
*   **Grader**: Programmatic, **deterministic** grading logic in `_evaluate()`.
*   **Metadata**: Verified `openenv.yaml`.

---

## 📜 Metadata
- **Version**: 1.0.0
- **Author**: Antigravity AI
- **Tags**: `data-cleaning`, `pandas`, `automation`, `real-world`
