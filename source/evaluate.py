from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from datasets import load_from_disk
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
import wandb
from PIL import Image

def evaluate_model(model_name, dataset_path, processor_name=None):
    # Initialize WANDB
    wandb.init(project="DI725_Phase2", config={"model": model_name})
    
    # Load model and processor
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_name)
    processor = PaliGemmaProcessor.from_pretrained(processor_name or model_name)
    
    # Load validation dataset
    dataset = load_from_disk(dataset_path)
    val_dataset = dataset.filter(lambda x: x['split'] == 'val')
    
    # Generate captions
    predictions = []
    references = []
    
    for sample in val_dataset:
        image = Image.open(sample['image']).convert('RGB')
        inputs = processor(text="Generate a caption:", images=image, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50)
        pred_caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        predictions.append(pred_caption)
        references.append([sample['caption']])  # List of reference captions
    
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
    wandb.log({
        "bleu_1": bleu_scores[0],
        "bleu_4": bleu_scores[3],
        "cider": cider_score,
        "rouge_l": rouge_score
    })
    
    print(f"BLEU-1: {bleu_scores[0]:.3f}, BLEU-4: {bleu_scores[3]:.3f}, CIDEr: {cider_score:.3f}, ROUGE-L: {rouge_score:.3f}")
    wandb.finish()

if __name__ == "__main__":
    # Evaluate baseline
    baseline_model = "google/paligemma-3b-mix-224"
    dataset_path = "./data/risc_processed"
    evaluate_model(baseline_model, dataset_path)
    
    # Evaluate LoRA model
    lora_model = "./models/paligemma_lora"
    evaluate_model(lora_model, dataset_path, processor_name=baseline_model)