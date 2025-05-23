import torch
import pandas as pd
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import LoraConfig, get_peft_model
from PIL import Image
import wandb
import os
from pathlib import Path


from huggingface_hub import login
login("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")  # paste your token directly here


def load_dataset(image_dir, caption_file, max_train_samples=1000, max_val_samples=500):
    df = pd.read_csv(caption_file)
    train_df = df[df['split'] == 'train'].head(max_train_samples)
    val_df = df[df['split'] == 'test'].head(max_val_samples)
    return train_df, val_df

def train_lora(model_name, image_dir, caption_file, output_dir, lora_rank=8, epochs=1, learning_rate=1e-5, max_train_samples=1000):
    wandb.init(project="DI725_Phase2", config={"lora_rank": lora_rank, "epochs": epochs, "lr": learning_rate})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model in float16 precision
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    train_df, val_df = load_dataset(image_dir, caption_file, max_train_samples, max_val_samples=500)
    print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for i, row in enumerate(train_df.itertuples()):
            try:
                image_path = os.path.join(image_dir, row.image)
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue
                image = Image.open(image_path).convert('RGB')
                
                caption = row.caption_1
                if pd.isna(caption):
                    print(f"Missing caption for image {row.image}")
                    continue
                text_input = f"<image> caption {caption}"
                
                inputs = processor(
                    text=text_input,
                    images=image,
                    return_tensors="pt",
                    padding="longest"
                )
                inputs = {k: v.to(device, dtype=torch.float16 if v.dtype == torch.float32 else None) for k, v in inputs.items()}
                
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    pixel_values=inputs['pixel_values'],
                    labels=inputs['input_ids']
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                if (i + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}, Step {i+1}, Loss: {loss.item():.4f}")
            except Exception as e:
                print(f"Error in sample {i+1}: {e}")
                continue
        
        avg_train_loss = total_loss / len(train_df)
        wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss})
        
        # Validation
        model.eval()
        val_loss = 0
        for i, row in enumerate(val_df.itertuples()):
            try:
                image_path = os.path.join(image_dir, row.image)
                if not os.path.exists(image_path):
                    continue
                image = Image.open(image_path).convert('RGB')
                caption = row.caption_1
                if pd.isna(caption):
                    continue
                text_input = f"<image> caption {caption}"
                inputs = processor(
                    text=text_input,
                    images=image,
                    return_tensors="pt",
                    padding="longest"
                )
                inputs = {k: v.to(device, dtype=torch.float16 if v.dtype == torch.float32 else None) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        pixel_values=inputs['pixel_values'],
                        labels=inputs['input_ids']
                    )
                    val_loss += outputs.loss.item()
            except Exception as e:
                print(f"Validation error at sample {i+1}: {e}")
                continue
        avg_val_loss = val_loss / len(val_df)
        wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss})
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")
        model.train()
    
    # Save model + processor
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    wandb.finish()

if __name__ == "__main__":
    model_name = "google/paligemma-3b-mix-224"
    image_dir = r"C:\Users\nesil.bor\Desktop\Folders\master\DI725\DI725_termproject_2030336\data\RISCM\resized"
    caption_file = r"C:\Users\nesil.bor\Desktop\Folders\master\DI725\DI725_termproject_2030336\data\RISCM\captions.csv"
    output_dir = r"C:\Users\nesil.bor\Desktop\Folders\master\DI725\DI725_termproject_2030336\source\models\paligemma_lora"
    
    train_lora(model_name, image_dir, caption_file, output_dir)



