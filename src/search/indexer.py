import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from src.model.dataset import ECommerceDataset, generate_synthetic_catalog
from src.model.model import ECommerceCLIP

def build_vector_index(catalog_path="data/catalog.json", checkpoint_path="data/checkpoints/best_model.pt", output_path="data/embeddings.npz", device=None):
    """
    Computes text and image embeddings for the entire product catalog
    and saves them as a compressed numpy array.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building vector index on device: {device.upper()}")

    # Ensure catalog exists
    if not os.path.exists(catalog_path):
        print("Catalog not found. Generating synthetic catalog...")
        generate_synthetic_catalog(output_dir=os.path.dirname(catalog_path))

    # Initialize CLIP model and processor
    model = ECommerceCLIP()
    
    # Load custom fine-tuned weights if checkpoint exists
    if os.path.exists(checkpoint_path):
        print(f"Loading fine-tuned checkpoint from '{checkpoint_path}'...")
        model.load_checkpoint(checkpoint_path, device=device)
    else:
        print("No fine-tuned checkpoint found. Initializing with pre-trained CLIP weights.")
        
    model.to(device)
    model.eval()
    
    # Load dataset
    dataset = ECommerceDataset(catalog_path=catalog_path)
    # Using batch size of 8 to prevent memory overflow on smaller GPUs
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    image_embeddings_list = []
    text_embeddings_list = []
    
    # Iterate through dataset and compute embeddings
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            
            # Compute embeddings
            img_embeds = model.get_image_embeddings(pixel_values, device=device)
            text_embeds = model.get_text_embeddings(input_ids, attention_mask, device=device)
            
            image_embeddings_list.append(img_embeds.cpu().numpy())
            text_embeddings_list.append(text_embeds.cpu().numpy())
            
    # Concatenate embeddings
    image_embeddings = np.vstack(image_embeddings_list)
    text_embeddings = np.vstack(text_embeddings_list)
    
    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path, 
        image_embeddings=image_embeddings, 
        text_embeddings=text_embeddings
    )
    
    print(f"Index built successfully. Saved image embeddings {image_embeddings.shape} "
          f"and text embeddings {text_embeddings.shape} to '{output_path}'.")
    return image_embeddings, text_embeddings

if __name__ == "__main__":
    build_vector_index()
