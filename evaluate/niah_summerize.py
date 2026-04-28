import os
import json


output_path = "evaluate/outputs/lm_eval_outputs/1B_niah_yarn_vllm"
model_names = [
    "1B_baseline",
    "1B_baseline_smoothquant",
    "1B_baseline_bnb",
    "1B_baseline_gptq",
    "1B_baseline_awq",
    "1B_vscale",
    "1B_vscale_smoothquant",
    "1B_vscale_bnb",
    "1B_vscale_gptq",
    "1B_vscale_awq",
]
task = "niah_single_2"
# task = "niah_single_3"
# task = "niah_multikey_1"
lengths = [1024, 2048, 4096, 8192]

summery_csv = "model"
latex_table = "model"
for length in lengths:
    summery_csv += f",{length}_mean,{length}_std"
    latex_table += f" & {length}"
summery_csv += "\n"
latex_table += " \\\\\n"

for model in model_names:
    summery_csv += model
    latex_table += model
    with open(f'{output_path}/{model}.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    for length in lengths:
        r = results[task][f"{length},none"]
        summery_csv += f",{r['mean']:.4f},{r['std']:.4f}"
        latex_table += f" & ${r['mean']*100:.2f}_" + "{\pm " + f"{r['std']*100:.2f}" + "}$"
    summery_csv += "\n"
    latex_table += " \\\\\n"

save_file = os.path.join(output_path, f"{task}_summary.csv")
save_table = os.path.join(output_path, f"{task}_table.tex")

with open(save_file, "w", encoding="utf-8") as f:
    f.write(summery_csv)
with open(save_table, "w", encoding="utf-8") as f:
    f.write(latex_table)
