from train_lora import train_lora
from evaluate import evaluate_model

def run_ablation_study():
    model_name = "google/paligemma-3b-mix-224"
    dataset_path = "./data/risc_processed"
    lora_ranks = [4, 8, 16]
    learning_rates = [1e-5, 5e-5]
    
    results = []
    
    for r in lora_ranks:
        for lr in learning_rates:
            output_dir = f"./models/paligemma_lora_r{r}_lr{lr}"
            print(f"Training with LoRA rank={r}, lr={lr}")
            train_lora(model_name, dataset_path, output_dir, lora_rank=r, epochs=3, learning_rate=lr)
            metrics = evaluate_model(output_dir, dataset_path, processor_name=model_name)
            results.append({
                'lora_rank': r,
                'learning_rate': lr,
                'bleu_1': metrics['bleu_1'],
                'bleu_4': metrics['bleu_4'],
                'cider': metrics['cider'],
                'rouge_l': metrics['rouge_l']
            })
    
    # Save results to CSV for report
    import pandas as pd
    pd.DataFrame(results).to_csv("./results/ablation_results.csv", index=False)

if __name__ == "__main__":
    run_ablation_study()