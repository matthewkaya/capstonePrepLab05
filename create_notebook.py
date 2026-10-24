"""Create the Lab 05 analysis notebook."""

from pathlib import Path

import nbformat as nbf

notebook = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell(
        "# Lab 05 — Predict Late Deliveries\n\n"
        "This notebook summarizes model selection, threshold "
        "optimization, and final business results."
    ),
    nbf.v4.new_code_cell(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "\n"
        "from latedeliveries.features import load_data\n"
        "\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' "
        "else Path.cwd()\n"
        "REPORTS = ROOT / 'reports'"
    ),
    nbf.v4.new_markdown_cell("## 1. Dataset Overview"),
    nbf.v4.new_code_cell(
        "data = load_data()\n"
        "print('Dataset shape:', data.shape)\n"
        "print('Late delivery rate:', round(data['late'].mean(), 4))\n"
        "data.head()"
    ),
    nbf.v4.new_markdown_cell("## 2. Model Comparison"),
    nbf.v4.new_code_cell(
        "cv_results = pd.read_csv(REPORTS / 'cv_results.csv')\n"
        "cv_results.round(4)"
    ),
    nbf.v4.new_markdown_cell(
        "## 3. Threshold Optimization\n\n"
        "The threshold was selected using out-of-fold training "
        "predictions, without using the test set."
    ),
    nbf.v4.new_code_cell(
        "threshold_results = pd.DataFrame(\n"
        "    json.loads((REPORTS / 'threshold_results.json').read_text())\n"
        ")\n"
        "ax = threshold_results.plot(\n"
        "    x='threshold', y='cost', marker='o', legend=False\n"
        ")\n"
        "ax.set_title('Training Cost by Classification Threshold')\n"
        "ax.set_xlabel('Threshold')\n"
        "ax.set_ylabel('Total Cost ($)')\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell("## 4. Final Test Evaluation"),
    nbf.v4.new_code_cell(
        "metrics = json.loads((REPORTS / 'metrics.json').read_text())\n"
        "pd.Series({\n"
        "    key: metrics[key]\n"
        "    for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']\n"
        "}).to_frame('value').round(4)"
    ),
    nbf.v4.new_code_cell(
        "cm = pd.DataFrame(\n"
        "    metrics['confusion_matrix'],\n"
        "    index=['Actual On Time', 'Actual Late'],\n"
        "    columns=['Predicted On Time', 'Predicted Late'],\n"
        ")\n"
        "cm"
    ),
    nbf.v4.new_markdown_cell("## 5. Business Cost Comparison"),
    nbf.v4.new_code_cell(
        "costs = pd.Series(metrics['costs'], name='Cost ($)')\n"
        "costs.to_frame()"
    ),
    nbf.v4.new_markdown_cell(
        "## 6. Recommendation\n\n"
        "Deploy the tuned logistic regression pipeline as a "
        "limited pilot using a 0.25 classification threshold. "
        "On the held-out test set, the model policy costs $1,443, "
        "compared with $1,755 when flagging no orders and $3,000 "
        "when flagging every order. This represents a 17.8% "
        "reduction relative to flagging none.\n\n"
        "The model uses only features assumed available at order "
        "placement. We exclude order_id, prep_minutes, and "
        "customer_rating to avoid identifiers and unavailable "
        "or leakage-prone information. Courier experience and "
        "vehicle type are assumed known at prediction time.\n\n"
        "The model detects only 41.0% of late orders. Results "
        "come from synthetic data, so operational performance "
        "may differ. Monitor costs, recall, and data drift "
        "before considering wider deployment."
    ),
]

notebook.cells = cells
notebook.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

output = Path("notebooks/analysis.ipynb")
output.parent.mkdir(parents=True, exist_ok=True)

nbf.write(notebook, output)

print(f"Created {output}")
print(f"Total cells: {len(cells)}")
