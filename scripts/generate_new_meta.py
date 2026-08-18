import os
import re
import sys
import yaml

MODEL_NAME = "akaruineko/ftan-2.5"
BENCHMARK_ID = "akaruineko/ftanch"

def parse_log_to_yaml(log_text):
    results = []
    pattern = re.compile(r"^\s*([a-zA-Z0-9_]+):\s*(.*)$", re.MULTILINE)

    for match in pattern.finditer(log_text):
        task_id = match.group(1)
        metrics_str = match.group(2)
        metric_pairs = re.findall(r"([a-zA-Z0-9_]+)=([0-9.]+)", metrics_str)
        for m_type, m_val in metric_pairs:
            if m_type == "f1": 
                results.append({
                    "dataset": {
                        "id": BENCHMARK_ID,
                        "task_id": task_id
                    },
                    "value": float(m_val) * 100  
                })

    return yaml.dump(results, allow_unicode=True, sort_keys=False, default_flow_style=False)

if __name__ == "__main__":
    print("Paste your log output below and press Ctrl+D (Linux) or Ctrl+Z (Windows):\n")
    log_output = sys.stdin.read()
    
    if not log_output.strip():
        print("Error: Input is empty.")
        sys.exit(1)
        
    generated_yaml = parse_log_to_yaml(log_output)
    
    os.makedirs(".eval_results", exist_ok=True)
    output_path = os.path.join(".eval_results", "ftanch_results.yaml")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(generated_yaml)
        
    print(f"\nSaved successfully to {output_path}! Push this folder to your model repository.")

