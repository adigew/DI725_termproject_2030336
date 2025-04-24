import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
import wandb

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

def train_lora(model_name, dataset_path, output_dir, lora_rank=8, epochs=3):
    # Initialize WANDB
    wandb.init(project="DI725_Phase2", config={"lora_rank": lora_rank, "epochs": epochs})
    
    # Load model and processor
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_name)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    
    for epoch in range(epochs):
        for batch in train_dataset.iter(batch_size=8):
            outputs = model(
                input_ids=batch['input_ids'].to(model.device),
                attention_mask=batch['attention_mask'].to(model.device),
                pixel_values=batch['pixel_values'].to(model.device),
                labels=batch['input_ids'].to(model.device)
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            wandb.log({"epoch": epoch, "train_loss": loss.item()})
        
        # Validation
        model.eval()
        val_loss = 0
        for batch in val_dataset.iter(batch_size=8):
            with torch.no_grad():
                outputs = model(
                    input_ids=batch['input_ids'].to(model.device),
                    attention_mask=batch['attention_mask'].to(model.device),
                    pixel_values=batch['pixel_values'].to(model.device),
                    labels=batch['input_ids'].to(model.device)
                )
                val_loss += outputs.loss.item()
        val_loss /= len(val_dataset)
        wandb.log({"epoch": epoch, "val_loss": val_loss})
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