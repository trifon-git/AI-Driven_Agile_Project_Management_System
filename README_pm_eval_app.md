# PM Evaluation App

The `pm_eval_app.py` script is a Streamlit web interface used to evaluate decomposition candidates.

## Prerequisites

Ensure you are in your Python environment and have the necessary installed packages:
```bash
pip install streamlit pandas numpy
```

## How to Run

1. Open your terminal.
2. Navigate to the root directory of the project (where the `scripts/` and `OUTPUT/` folders are located).
3. Start the Streamlit server by running:
   ```bash
   streamlit run scripts/pm_eval_app.py
   ```
4. The application will automatically open in your default web browser (typically at `http://localhost:8501`).

*Note: The script expects the input files to be located in `OUTPUT/eval_candidates/` and will save your evaluations to `OUTPUT/eval_project_managment/`.*
