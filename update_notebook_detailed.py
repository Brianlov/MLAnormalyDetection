import json

notebook_path = 'comparative_analysis.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Imports cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'from sklearn.metrics import' in source:
             if 'classification_report' not in source:
                 # Be careful not to double add if it was already modified differently
                 # We'll just replace the line
                 new_lines = []
                 for line in cell['source']:
                     if 'from sklearn.metrics import' in line:
                         line = line.strip()
                         if 'classification_report' not in line:
                            line += ', classification_report\n'
                         else:
                            line += '\n'
                         new_lines.append(line)
                     else:
                         new_lines.append(line)
                 cell['source'] = new_lines
             break

# 2. Update the Results Cell
# We look for the cell that iterates over models.items() and prints the table
target_string = "for name, preds in models.items():"

new_code_block = [
    "# Print Comparison Table\n",
    "print(f\"{'Model':<30} | {'F1':<6} | {'Prec':<6} | {'Rec':<6} | {'Acc':<6} | {'TN':<4} {'FP':<4} {'FN':<4} {'TP':<4}\")\n",
    "print(\"-\" * 95)\n",
    "\n",
    "results = []\n",
    "for name, preds in models.items():\n",
    "    # Summary Table Metrics\n",
    "    f1 = f1_score(df_total['ground_truth'], preds)\n",
    "    prec = precision_score(df_total['ground_truth'], preds)\n",
    "    rec = recall_score(df_total['ground_truth'], preds)\n",
    "    acc = np.mean(preds == df_total['ground_truth'])\n",
    "    tn, fp, fn, tp = confusion_matrix(df_total['ground_truth'], preds).ravel()\n",
    "    \n",
    "    results.append({'Model': name, 'F1': f1, 'Precision': prec, 'Recall': rec, 'Accuracy': acc, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})\n",
    "    print(f\"{name:<30} | {f1:.3f}  | {prec:.3f}  | {rec:.3f}  | {acc:.3f}  | {tn:<4} {fp:<4} {fn:<4} {tp:<4}\")\n",
    "\n",
    "# Print Detailed Reports\n",
    "print(\"\\n\" + \"=\"*40)\n",
    "print(\"DETAILED CLASSIFICATION REPORTS\")\n",
    "print(\"=\"*40)\n",
    "\n",
    "for name, preds in models.items():\n",
    "    print(f\"\\n--- {name} Evaluation ---\")\n",
    "    print(classification_report(df_total['ground_truth'], preds))\n",
    "    print(\"Confusion Matrix:\")\n",
    "    print(confusion_matrix(df_total['ground_truth'], preds))\n",
    "    print(\"-\" * 40)"
]

updated = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # Identify the correct cell by looking for the unique header or loop structure
        if target_string in source and "f1_score" in source:
             # We assume models dict is defined in this cell or previous. 
             # If defined here, we need to keep it.
             # In the previous update, we saw `results = []` as a split point.
             
             parts = source.split("results = []")
             if len(parts) > 1:
                 top_part = parts[0] # Includes models = {...}
                 # Reconstruct: Top part + New code block (which starts with results = [])
                 # We skip the first line of new_code_block because it's a comment for clarity, 
                 # but actually let's just replace everything after the models definition to be safe.
                 
                 # Actually, my new_block starts with `# Print Comparison Table`. 
                 # The user's cell might have `models = ` at the top.
                 
                 cell['source'] = [l for l in top_part.splitlines(keepends=True)] + new_code_block
                 updated = True
                 break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Notebook updated: {updated}")
