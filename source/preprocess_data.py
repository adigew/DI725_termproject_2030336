from datasets import Dataset, concatenate_datasets
import pandas as pd
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

def load_risc_dataset(data_dir, captions_file):
    # Load captions
    captions_df = pd.read_csv(captions_file)
    print(f"Loaded {len(captions_df)} rows from captions file.")
    
    # Check available caption columns (caption_1, caption_2, etc.)
    caption_columns = [col for col in captions_df.columns if col.startswith('caption_')]
    
    # Use caption_1 as the main caption and create the 'caption' column
    captions_df['caption'] = captions_df['caption_1'].str.replace(r'\b(white|green)\b', '', regex=True).str.strip()
    
    # Create dataset dictionary
    dataset_dict = {
        'image': [],
        'caption': [],
        'split': []
    }
    
    for _, row in captions_df.iterrows():
        # Check if the image filename already has .jpg extension
        image_filename = row['image']
        if not image_filename.lower().endswith('.jpg'):
            image_filename += '.jpg'
        img_path = os.path.join(data_dir, image_filename)
        if os.path.exists(img_path):
            dataset_dict['image'].append(img_path)
            dataset_dict['caption'].append(row['caption'])
            dataset_dict['split'].append(row['split'])
        else:
            print(f"Image not found: {img_path}")
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict(dataset_dict)
    print(f"Created dataset with {len(dataset)} examples.")
    return dataset

def augment_captions(dataset, max_samples=5000):
    # Load BLIP for caption augmentation
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    augmented_data = {'image': [], 'caption': [], 'split': []}
    
    # Augment only training data
    train_data = dataset.filter(lambda x: x['split'] == 'train')
    print(f"Found {len(train_data)} training examples to augment.")
    
    for i, sample in enumerate(train_data):
        if i >= max_samples:
            break
        image = Image.open(sample['image']).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        augmented_data['image'].append(sample['image'])
        augmented_data['caption'].append(caption)
        augmented_data['split'].append('train')
    
    # Combine original and augmented data
    augmented_dataset = Dataset.from_dict(augmented_data)
    print(f"Created augmented dataset with {len(augmented_dataset)} examples.")
    
    final_dataset = concatenate_datasets([dataset, augmented_dataset])
    print(f"Final dataset size after concatenation: {len(final_dataset)} examples.")
    return final_dataset

if __name__ == "__main__":
    data_dir = r"C:\Users\nesil.bor\Desktop\Folders\master\DI725\DI725_termproject_2030336\data\RISCM\resized"
    captions_file = r"C:\Users\nesil.bor\Desktop\Folders\master\DI725\DI725_termproject_2030336\data\RISCM\captions.csv"
    dataset = load_risc_dataset(data_dir, captions_file)
    
    # Augment training data
    augmented_dataset = augment_captions(dataset)
    
    # Save or use dataset
    augmented_dataset.save_to_disk("./data/risc_processed")
    
    
from datasets import load_from_disk
saved_dataset = load_from_disk("./data/risc_processed")
for i in range(5):
    print(saved_dataset[-5+i]['caption'])  # Print the last 5 captions (likely augmented ones)