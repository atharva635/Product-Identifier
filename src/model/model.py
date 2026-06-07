# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from transformers import CLIPModel

class ECommerceCLIP(nn.Module):
    """
    Wrapper for Hugging Face CLIP model to generate normalized embeddings
    and support fine-tuning contrastive alignment.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask, pixel_values):
        """
        Forward pass during training. Uses CLIP's built-in contrastive loss
        calculated from the aligned logits.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
            return_loss=True
        )
        # return loss and logits
        return outputs.loss, outputs.logits_per_image, outputs.logits_per_text

    def get_image_embeddings(self, pixel_values, device="cpu"):
        """Extracts and L2-normalizes visual embeddings."""
        self.model.eval()
        self.model.to(device)
        pixel_values = pixel_values.to(device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(pixel_values=pixel_values)
            if not isinstance(image_features, torch.Tensor):
                image_features = image_features.pooler_output
            # Apply L2 normalization to project to unit hypersphere
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        return image_features

    def get_text_embeddings(self, input_ids, attention_mask, device="cpu"):
        """Extracts and L2-normalizes textual embeddings."""
        self.model.eval()
        self.model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            if not isinstance(text_features, torch.Tensor):
                text_features = text_features.pooler_output
            # Apply L2 normalization
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return text_features

    def save_checkpoint(self, path):
        """Saves the underlying model state dict."""
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path, device="cpu"):
        """Loads a saved checkpoint into the model wrapper."""
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
