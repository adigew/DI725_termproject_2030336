import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(baseline_metrics, lora_metrics, output_path):
    metrics = ['BLEU-1', 'BLEU-4', 'CIDEr', 'ROUGE-L']
    baseline_scores = [baseline_metrics[m.lower()] for m in metrics]
    lora_scores = [lora_metrics[m.lower()] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, baseline_scores, width, label='Baseline', hatch='//')
    plt.bar(x + width/2, lora_scores, width, label='LoRA', hatch='xx')
    
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Baseline vs. LoRA Fine-Tuning Performance')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, linestyle='--')
    
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Example metrics (replace with actual results from evaluate.py)
    baseline_metrics = {'bleu_1': 0.60, 'bleu_4': 0.25, 'cider': 0.80, 'rouge_l': 0.45}
    lora_metrics = {'bleu_1': 0.63, 'bleu_4': 0.28, 'cider': 0.87, 'rouge_l': 0.48}
    plot_metrics(baseline_metrics, lora_metrics, "./figures/metrics_comparison.png")