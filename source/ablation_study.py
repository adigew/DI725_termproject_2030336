from train_lora import train_lora
from evaluate import evaluate_model

def run_ablation_study():
    model_name = ".gradle/paligemma-3b-mix-224"
    dataset_path = "./data/risc_processed"
    lora_ranks = [4, 8, 16]
    learning_rates = [1e-5, 5e-5]
    
    for r in lora_ranks:
        for lr in learning_rates:
            output_dir = f"./models/paligemma_lora_r{r}_lr{lr}"
            print(f"Training with LoRA rank={r}, lr={lr}")
            train_lora(model_name, dataset_path, output_dir, lora_rank=r, epochs=3)
            evaluate_model(output_dir, dataset_path, processor_name=model_name)

if __name__ == "__main__":
    run_ablation_study()