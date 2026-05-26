<div align="center">
  <h1>🚀 Antigrivity: AI Agile PM Helper</h1>
  <p><i>An agentic workflow system for automating and optimizing Agile project management using Local LLMs, Graph Neural Networks, and ML.</i></p>

  ![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python)
  ![Ollama](https://img.shields.io/badge/Ollama-Offline%20LLMs-lightgrey.svg?style=for-the-badge&logo=ollama)
  ![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg?style=for-the-badge&logo=streamlit)
  ![PyTorch](https://img.shields.io/badge/PyTorch-Node2Vec-EE4C2C.svg?style=for-the-badge&logo=pytorch)
</div>

---

**Antigrivity** uses multi-agent orchestration to automatically decompose high-level project descriptions into atomic tasks, estimate developer hours, and intelligently allocate work to team members based on capacity, skills, and historical data patterns.

## 🌟 Key Features
- **🧠 Agentic Decomposition**: Breaks down single-sentence project requests into fully fleshed-out Jira-style tasks.
- **⏱️ ML-Based Estimation**: Predicts required developer hours using historical completion data via Scikit-Learn regressors.
- **🕸️ GNN Allocation (Node2Vec)**: Uses NetworkX and PyTorch to model historical domain overlaps between developers and tasks as a graph, ensuring optimal sprint allocations.
- **🔒 Fully Private**: Powered entirely by local models via **Ollama**, ensuring your proprietary project data never leaves your infrastructure.
- **🎨 Interactive Web UI**: A beautiful, user-friendly Streamlit interface to manage your PM pipelines.

## 🚀 Technologies Used
* **Python 3.9+**: Core system architecture.
* **Ollama (`qwen3.6:27b`)**: Local LLM engine for offline task decomposition. 
* **Graph Neural Networks (Node2Vec)**: Leverages `networkx` and `torch` to model organizational capacity and developer workflows.
* **SentenceTransformers (`intfloat/multilingual-e5-large`)**: Creates vector embeddings of task descriptions to calculate semantic alignment to developer skill profiles.
* **Scikit-Learn**: Trains the task duration estimator.
* **Streamlit**: Powers the interactive web console.
* **NetworkX & PyVis**: Builds and visualizes complex sprint allocation graphs.

---

## 📁 Project Structure

```text
antigrivity/        # Core Application
  ├── core/         # Data classes + Orchestrator
  ├── agents/       # AI agents (Decomposition, Estimation, Allocation)
  └── models/       # ML Models (Node2Vec embedder, Estimators, Embedding matchers)
scripts/            # CLI scripts for preprocessing, training, and evaluation
DATA/               # Raw & Processed datasets
MODELS/             # Saved ML model state files (.pkl, .pt)
OUTPUT/             # Generated HTML visuals, Sprint plans and Eval Candidates
```

---

## 🛠️ Setup & Installation

### 1. Python Environment
Clone the repository and install the dependencies:
```bash
git clone https://github.com/yourusername/antigrivity.git](https://github.com/trifon-git/AI-Driven_Agile_Project_Management_System
pip install -r requirements.txt
```

### 2. Install & Configure Ollama
The Decomposition Agent relies on local LLMs to break down features.
1. Install [Ollama](https://ollama.com/) for your OS.
2. Pull your preferred models (default configured model is `qwen3.6:27b`):
```bash
ollama run gemma3:12b
ollama run llama3.2:latest
```
*(If using a different model, set the `OLLAMA_MODEL` variable in `antigrivity/config.py`)*

---

## 📊 Data Preprocessing & Model Training
Before utilizing the intelligent GNN allocation or time estimations, you must establish the baseline models using your historical Jira datasets. 

From the project root, run the setup pipeline:

1. **Merge Data**: Combines Jira exports mapped to Timesheet hours.
   ```bash
   python scripts/01_merge_data.py
   ```
2. **Feature Engineering**: Cleans text and extracts statistical features.
   ```bash
   python scripts/02_feature_engineering.py
   ```
3. **Train Predictor**: Trains scikit-learn models for task hours approximations.
   ```bash
   python scripts/03_train_v2.py
   ```
4. **Build Task Embeddings**: Generates NLP embeddings for semantic matching.
   ```bash
   python scripts/build_embeddings.py
   ```
5. **Graph Preprocessing**: Builds the NetworkX graph and Node2Vec embeddings for the GNN Allocator.
   ```bash
   python scripts/preprocess_graph.py
   ```

---

## ⚙️ Running the Agentic Pipeline

Generate a full sprint plan from a single sentence! The Orchestrator handles passing context down the line: **Decomposition ➡️ Estimation ➡️ Allocation (GNN)**.

### Web UI (Recommended)
Launch the interactive Streamlit portal:
```bash
streamlit run antigrivity/app.py
```

### CLI Orchestrator
**Demographic/Automated Usage:**
```bash
python -m antigrivity.run_pipeline --demo
```

**Custom Projects:**
```bash
python -m antigrivity.run_pipeline --desc "Build a React.js e-commerce checkout page with Stripe integration." --project-key "ECOMM"
```

**Interactive CLI Mode:**
*(Allows PM to dictate constraints to the AI loop to re-optimize allocations)*
```bash
python -m antigrivity.run_pipeline --demo --interactive
```

*(You can toggle back to traditional heuristic assignment by setting `USE_GNN_ALLOCATION = False` in `antigrivity/config.py`)*

---

## 📝 Benchmark Evaluation Framework

Use our custom Benchmark Evaluation toolkit to quantitatively compare how the AI models perform against your human Project Managers.

### 1. Generate AI Baselines
Run predefined projects through multiple LLMs and Allocation strategies to generate options.
```bash
python scripts/generate_benchmarks.py
```
*Locate `decomposition_eval.csv` and `allocation_eval.csv` in `OUTPUT/eval_candidates/`.*

### 2. PM Ground Truth Scoring
1. Open the generated CSVs.
2. Under **Decomposition CSV**: Rate the AI (`PM Score (1-5)` & `Completeness (1-5)`).
3. Under **Allocation CSV**: Add your target dev in the `PM Ground Truth Assignee` column.

### 3. Analyze Accuracy Metrics
Parse the CSV scores to compute overall AI vs PM alignment over the GNN vs Heuristic engines.
```bash
python scripts/analyze_evaluations.py
```

---

<p align="center"><i>Building the Future of Autonomous Project Management 🛠️</i></p>
