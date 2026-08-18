import re
import sys
import yaml

MODEL_NAME = "akaruineko/ftan-2.0-final"
BENCHMARK_ID = "akaruineko/ftanch"
BENCHMARK_NAME = "FTANch"

METRIC_NAMES = {
    "acc": "Accuracy",
    "p": "Precision",
    "r": "Recall",
    "f1": "F1-Score",
}

def parse_log_to_yaml(log_text):
    results = []
    pattern = re.compile(r"^\s*([a-zA-Z0-9_]+):\s*(.*)$", re.MULTILINE)

    for match in pattern.finditer(log_text):
        task_id = match.group(1)
        metrics_str = match.group(2)
        metric_pairs = re.findall(r"([a-zA-Z0-9_]+)=([0-9.]+)", metrics_str)

        hf_metrics = []
        for m_type, m_val in metric_pairs:
            hf_metrics.append(
                {
                    "type": m_type,
                    "value": float(m_val),
                    "name": METRIC_NAMES.get(m_type, m_type),
                }
            )

        task_block = {
            "task": {"type": "text-classification", "id": task_id},
            "dataset": {
                "type": BENCHMARK_ID,
                "name": BENCHMARK_NAME,
                "split": "test",
            },
            "metrics": hf_metrics,
        }
        results.append(task_block)

    final_meta = {"model-index": [{"name": MODEL_NAME, "results": results}]}
    yaml_text = yaml.dump(
        final_meta, allow_unicode=True, sort_keys=False, default_flow_style=False
    )
    return f"---\n{yaml_text}---"

if __name__ == "__main__":
    print("Paste your log output below and press Ctrl+D (Linux/Mac) or Ctrl+Z (Windows) when done:\n")
    log_output = sys.stdin.read()
    
    if not log_output.strip():
        print("Error: Input is empty.")
        sys.exit(1)
        
    generated_yaml = parse_log_to_yaml(log_output)
    print("\nGenerated Hugging Face Metadata:\n")
    print(generated_yaml)

    with open("hf_metadata.yml", "w", encoding="utf-8") as f:
        f.write(generated_yaml)
    print("\nSaved to hf_metadata.yml")

