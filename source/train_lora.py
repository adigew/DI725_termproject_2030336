import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
import wandb
from PIL import Image

def prepare_dataset(dataset, processor):
    def preprocess(example):
        image = Image.open(example['image']).convert('RGB')
        inputs = processor(text=example['caption'], images=image, return_tensors="pt", padding="longest")
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze()
        }
    return dataset.map(preprocess, remove_columns=['image', 'caption', 'split'])

def train_lora(model_name, dataset_path, output_dir, lora_rank=8, epochs=3, learning_rate=1e-5):
    # Initialize WANDB
    wandb.init(project="DI725_Phase2", config={"lora_rank": lora_rank, "epochs": epochs, "lr": learning_rate})
    
    # Load model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_name).to(device)
    processor = PaliGemmaProcessor.from_pretrained(model_name)
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Adjust based on PaliGemma architecture
        lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    train_dataset = prepare_dataset(dataset.filter(lambda x: x['split'] == 'train'), processor)
    val_dataset = prepare_dataset(dataset.filter(lambda x: x['split'] == 'val'), processor)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(train_dataset.iter(batch_size=8)):
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                pixel_values=batch['pixel_values'].to(device),
                labels=batch['input_ids'].to(device)
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Step {i+1}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_dataset)
        wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss})
        
        # Validation
        model.eval()
        val_loss = 0
        for batch in val_dataset.iter(batch_size=8):
            with torch.no_grad():
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    pixel_values=batch['pixel_values'].to(device),
                    labels=batch['input_ids'].to(device)
                )
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(val_dataset)
        wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss})
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")
        model.train()
    
    # Save model
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    wandb.finish()

if __name__ == "__main__":
    model_name = "google/paligemma-3b-mix-224"
    dataset_path = "./data/risc_processed"
    output_dir = "./models/paligemma_lora"
    train_lora(model_name, dataset_path, output_dir)