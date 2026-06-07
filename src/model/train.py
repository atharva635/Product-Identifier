import os
import json
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from src.model.dataset import ECommerceDataset, generate_synthetic_catalog
from src.model.model import ECommerceCLIP

def calculate_metrics(logits_per_image, logits_per_text):
    """Calculates top-1 and top-5 retrieval accuracy from similarity logits."""
    # logits_per_image shape: [B, B]
    # Ground truth is the diagonal index (since batch items are aligned)
    batch_size = logits_per_image.shape[0]
    targets = torch.arange(batch_size, dtype=torch.long, device=logits_per_image.device)
    
    k = min(5, batch_size)
    
    # Image-to-Text retrieval accuracy
    _, i2t_preds = logits_per_image.topk(k, dim=-1)
    i2t_acc1 = (i2t_preds[:, 0] == targets).float().mean().item()
    i2t_acc5 = (i2t_preds == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
    
    # Text-to-Image retrieval accuracy
    _, t2i_preds = logits_per_text.topk(k, dim=-1)
    t2i_acc1 = (t2i_preds[:, 0] == targets).float().mean().item()
    t2i_acc5 = (t2i_preds == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
    
    return {
        "i2t_acc1": i2t_acc1,
        "i2t_acc5": i2t_acc5,
        "t2i_acc1": t2i_acc1,
        "t2i_acc5": t2i_acc5,
        "mean_acc1": (i2t_acc1 + t2i_acc1) / 2.0
    }

def train_clip(epochs=5, batch_size=8, lr=5e-6, catalog_path="data/catalog.json", output_dir="data"):
    # Create output directories
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device.upper()}")
    
    # Generate catalog if it does not exist
    if not os.path.exists(catalog_path):
        print("Catalog not found. Generating synthetic catalog...")
        generate_synthetic_catalog(output_dir=os.path.dirname(catalog_path))
        
    # Load dataset
    dataset = ECommerceDataset(catalog_path=catalog_path)
    
    # Train / Val Split (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Load Model
    print("Loading pre-trained CLIP model (openai/clip-vit-base-patch32)...")
    model = ECommerceCLIP()
    model.to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Training History Log
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc1": [],
        "val_acc1": [],
        "train_acc5": [],
        "val_acc5": []
    }
    
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        # --- TRAINING ---
        model.train()
        total_train_loss = 0
        train_i2t_logits_list = []
        train_t2i_logits_list = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            
            loss, logits_per_image, logits_per_text = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_i2t_logits_list.append(logits_per_image.detach())
            train_t2i_logits_list.append(logits_per_text.detach())
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Calculate training metrics on the last batch of the epoch
        train_metrics = calculate_metrics(train_i2t_logits_list[-1], train_t2i_logits_list[-1])
        
        # --- VALIDATION ---
        model.eval()
        total_val_loss = 0
        val_i2t_logits_list = []
        val_t2i_logits_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                
                loss, logits_per_image, logits_per_text = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values
                )
                
                total_val_loss += loss.item()
                val_i2t_logits_list.append(logits_per_image)
                val_t2i_logits_list.append(logits_per_text)
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Calculate validation metrics on the last validation batch
        val_metrics = calculate_metrics(val_i2t_logits_list[-1], val_t2i_logits_list[-1])
        
        # Log stats
        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val Top-1 Acc: {val_metrics['mean_acc1'] * 100:.1f}%")
        
        # Save checkpoints
        checkpoint_path = os.path.join(checkpoint_dir, "latest_model.pt")
        model.save_checkpoint(checkpoint_path)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            model.save_checkpoint(best_path)
            print(f"--> Saved new best checkpoint to '{best_path}'")
            
        # Record history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc1"].append(train_metrics["mean_acc1"])
        history["val_acc1"].append(val_metrics["mean_acc1"])
        history["train_acc5"].append((train_metrics["i2t_acc5"] + train_metrics["t2i_acc5"]) / 2.0)
        history["val_acc5"].append((val_metrics["i2t_acc5"] + val_metrics["t2i_acc5"]) / 2.0)
        
    # Write history to json
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
        
    print(f"Training completed. History log saved to '{history_path}'.")
    return history

if __name__ == "__main__":
    train_clip(epochs=3)
