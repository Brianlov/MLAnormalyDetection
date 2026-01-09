import json

notebook_path = 'comparative_analysis.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The code to replace
target_code_snippet = """for name, preds in models.items():
    f1 = f1_score(df_total['ground_truth'], preds)
    prec = precision_score(df_total['ground_truth'], preds)
    rec = recall_score(df_total['ground_truth'], preds)
    acc = np.mean(preds == df_total['ground_truth'])
    
    results.append({'Model': name, 'F1': f1, 'Precision': prec, 'Recall': rec, 'Accuracy': acc})
    print(f"{name:<30} | {f1:.3f}  | {prec:.3f}     | {rec:.3f}  | {acc:.3f}")"""

# The new code
new_code = """results = []
print(f"{'Model':<30} | {'F1':<6} | {'Prec':<6} | {'Rec':<6} | {'Acc':<6} | {'TN':<4} {'FP':<4} {'FN':<4} {'TP':<4}")
print("-" * 95)

for name, preds in models.items():
    f1 = f1_score(df_total['ground_truth'], preds)
    prec = precision_score(df_total['ground_truth'], preds)
    rec = recall_score(df_total['ground_truth'], preds)
    acc = np.mean(preds == df_total['ground_truth'])
    tn, fp, fn, tp = confusion_matrix(df_total['ground_truth'], preds).ravel()
    
    results.append({'Model': name, 'F1': f1, 'Precision': prec, 'Recall': rec, 'Accuracy': acc, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})
    print(f"{name:<30} | {f1:.3f}  | {prec:.3f}  | {rec:.3f}  | {acc:.3f}  | {tn:<4} {fp:<4} {fn:<4} {tp:<4}")"""

updated = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if target_code_snippet in source_str:
            cell['source'] = new_code.splitlines(keepends=True)
            updated = True
            # Also update cell directly above which likely contains the header or 'results = []' initialization if separate
            # But here the target snippet contains the loop, so we replace the whole block if it matches.
            # Actually, looking at the previous file view, the `results = []` and print header are in the same cell as the loop.
            # My target_code_snippet didn't include the header initially, let's adjust logic.
            
            # Better approach: Look for cell containing "for name, preds in models.items():"
            # and replace the entire content of that cell with our full new block + model definition if needed.
            # But the user provided snippet starts with `models = ...`.
            # Let's just find the cell that has the loop and replace the whole cell content to be safe.
            pass

# Let's act on the cell containing the loop
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "for name, preds in models.items():" in source and "f1_score" in source:
             # This is the evaluation cell. 
             # We will keep the 'models = {...}' part if present, and replace the loop part.
             
             # Split at 'results = []' since that resets the list
             parts = source.split("results = []")
             if len(parts) > 1:
                 # Reconstruct top part (models dict) + new bottom part
                 top_part = parts[0]
                 new_bottom = new_code # starts with results = []
                 cell['source'] = (top_part + new_bottom).splitlines(keepends=True)
                 updated = True
                 break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Notebook update status: {updated}")
