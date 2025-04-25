from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from datasets import load_from_disk
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
import wandb
from PIL import Image
import torch

def evaluate_model(model_name, dataset_path, processor_name=None):
    # Initialize WANDB
    wandb.init(project="DI725_Phase2", config={"model": model_name})
    
    # Load model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_name).to(device)
    processor = PaliGemmaProcessor.from_pretrained(processor_name or model_name)
    
    # Load validation dataset
    dataset = load_from_disk(dataset_path)
    val_dataset = dataset.filter(lambda x: x['split'] == 'val')
    
    # Generate captions
    predictions = []
    references = []
    
    model.eval()
    for sample in val_dataset:
        image = Image.open(sample['image']).convert('RGB')
        inputs = processor(text="Generate a caption:", images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        pred_caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        predictions.append(pred_caption)
        references.append([sample['caption']])
    
    # Compute metrics
    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score({i: ref for i, ref in enumerate(references)},
                                               {i: [pred] for i, pred in enumerate(predictions)})
    
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score({i: ref for i, ref in enumerate(references)},
                                                {i: [pred] for i, pred in enumerate(predictions)})
    
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score({i: ref for i, ref in enumerate(references)},
                                                {i: [pred] for i, pred in enumerate(predictions)})
    
    # Log results
    results = {
        "bleu_1": bleu_scores[0],
        "bleu_4": bleu_scores[3],
        "cider": cider_score,
        "rouge_l": rouge_score
    }
    wandb.log(results)
    print(f"Model: {model_name}")
    print(f"BLEU-1: {bleu_scores[0]:.3f}, BLEU-4: {bleu_scores[3]:.3f}, CIDEr: {cider_score:.3f}, ROUGE-L: {rouge_score:.3f}")
    wandb.finish()
    
    return results

if __name__ == "__main__":
    dataset_path = "./data/risc_processed"
    
    # Evaluate baseline
    baseline_model = "google/paligemma-3b-mix-224"
    baseline_results = evaluate_model(baseline_model, dataset_path)
    
    # Evaluate LoRA model
    lora_model = "./models/paligemma_lora"
    lora_results = evaluate_model(lora_model, dataset_path, processor_name=baseline_model)