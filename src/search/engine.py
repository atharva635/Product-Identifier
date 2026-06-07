import os
import json
import re
import numpy as np
from PIL import Image
import torch
from rank_bm25 import BM25Okapi
from transformers import CLIPProcessor

class HybridSearchEngine:
    """
    Hybrid search engine combining lexical (BM25) and semantic (CLIP dense vector) search.
    Supports price/brand parsing, multimodal search, recommendations, and Explainable AI.
    """
    def __init__(self, catalog_path="data/catalog.json", embeddings_path="data/embeddings.npz", processor_name="openai/clip-vit-base-patch32"):
        self.catalog_path = catalog_path
        self.embeddings_path = embeddings_path
        self.processor_name = processor_name
        
        self.catalog = []
        self.image_embeddings = None
        self.text_embeddings = None
        self.bm25 = None
        self.tokenized_corpus = []
        self.processor = None

    def load(self):
        """Loads catalog metadata and pre-computed embeddings."""
        if not os.path.exists(self.catalog_path):
            raise FileNotFoundError(f"Catalog file not found at {self.catalog_path}. Please build index first.")
            
        with open(self.catalog_path, "r") as f:
            self.catalog = json.load(f)
            
        if os.path.exists(self.embeddings_path):
            data = np.load(self.embeddings_path)
            self.image_embeddings = data["image_embeddings"]
            self.text_embeddings = data["text_embeddings"]
            print(f"Loaded embeddings: Image {self.image_embeddings.shape}, Text {self.text_embeddings.shape}")
        else:
            print("Embeddings file not found. Search will fallback to BM25-only until indexed.")
            
        # Initialize BM25 search index
        self._initialize_bm25()
        
        # Load processor lazily
        self.processor = CLIPProcessor.from_pretrained(self.processor_name)

    def _initialize_bm25(self):
        """Prepares catalog descriptions and titles for BM25 lexical search."""
        self.tokenized_corpus = []
        for item in self.catalog:
            # Tokenize title, description, brand, and tags for rich lexical indices
            text = f"{item['title']} {item['description']} {item['brand']} {' '.join(item['tags'])}".lower()
            tokens = [w for w in text.split() if len(w) > 1]
            self.tokenized_corpus.append(tokens)
            
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def parse_filters(self, query_text):
        """
        Parses metadata constraints from query text (e.g. 'under 3000', 'below Rs. 50000').
        Returns the parsed constraints and the cleaned query text.
        """
        cleaned_query = query_text.lower()
        max_price = None
        min_price = None
        
        # Match 'under/below/less than <number>'
        under_match = re.search(r'(?:under|below|less\s+than)\s*(?:rs\.?|inr|₹)?\s*([0-9,]+)', cleaned_query)
        if under_match:
            try:
                price_str = under_match.group(1).replace(',', '')
                max_price = int(price_str)
                # Remove the parsed segment from the query to avoid distracting the CLIP model
                cleaned_query = cleaned_query.replace(under_match.group(0), "")
            except Exception:
                pass

        # Match 'above/over/more than <number>'
        above_match = re.search(r'(?:above|over|more\s+than|greater\s+than)\s*(?:rs\.?|inr|₹)?\s*([0-9,]+)', cleaned_query)
        if above_match:
            try:
                price_str = above_match.group(1).replace(',', '')
                min_price = int(price_str)
                cleaned_query = cleaned_query.replace(above_match.group(0), "")
            except Exception:
                pass
                
        # Clean double spaces
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
        if not cleaned_query:
            cleaned_query = query_text # Fallback if everything was stripped

        return cleaned_query, min_price, max_price

    def get_explainability(self, item, query_text, dense_score, sparse_score):
        """Generates list of human-readable matching explanations for Recruiters (Explainable AI)."""
        reasons = []
        q = query_text.lower()
        
        # 1. Brand match
        if item["brand"].lower() in q:
            reasons.append(f"Brand Align: matches brand '{item['brand']}' specified in search")
            
        # 2. Color match
        if item["color"].lower() in q:
            reasons.append(f"Color Align: matches requested color '{item['color']}'")
            
        # 3. Category match
        if item["category"].lower() in q or (item["category"] == "Footwear" and ("shoe" in q or "sneaker" in q or "boot" in q or "sandal" in q)):
            reasons.append(f"Category Match: aligned with '{item['category']}' category")
            
        # 4. Tags match
        matched_tags = [tag for tag in item["tags"] if tag.lower() in q]
        if matched_tags:
            reasons.append(f"Attribute Match: matches keyword/tag(s): {', '.join(matched_tags)}")

        # 5. Semantic similarity explanation
        if dense_score > 0.8:
            reasons.append("High Semantic Relevance: strong visual-textual context match via fine-tuned CLIP")
        elif dense_score > 0.65:
            reasons.append("Medium Semantic Relevance: contextual association detected by CLIP")
            
        # 6. Lexical keyword match
        if sparse_score > 0.4:
            reasons.append("Direct Keyword Match: title/description text matches search keywords closely")

        # Fallback if no matching reasons
        if not reasons:
            reasons.append("General Match: retrieved based on global semantic context similarity")
            
        return reasons

    def search_by_text(self, model, query_text, hybrid_weight=0.5, top_k=6, device="cpu"):
        """
        Performs hybrid (Lexical BM25 + Semantic Vector) search with filter parsing and Explainable AI.
        """
        # Ensure metadata is loaded
        if not self.catalog:
            self.load()
            
        # 1. Parse query constraints (price limits)
        semantic_query, min_price, max_price = self.parse_filters(query_text)
        
        # Apply filters to catalog
        filtered_indices = []
        for idx, item in enumerate(self.catalog):
            keep = True
            if min_price is not None and item["price"] < min_price:
                keep = False
            if max_price is not None and item["price"] > max_price:
                keep = False
            if keep:
                filtered_indices.append(idx)
                
        # If filters are too restrictive and nothing matches, fall back to unfiltered to provide a better UX
        if not filtered_indices:
            filtered_indices = list(range(len(self.catalog)))
            max_price = None # reset warning/filter
            min_price = None
            
        # 2. Semantic (Dense Vector) Search on filtered subset
        dense_scores = np.zeros(len(self.catalog))
        if self.text_embeddings is not None and model is not None:
            inputs = self.processor(text=[semantic_query], return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Extract query embedding
            query_embedding = model.get_text_embeddings(input_ids, attention_mask, device=device)
            query_embedding = query_embedding.cpu().numpy().squeeze(0) # [dim]
            
            # Dot product for cosine similarity
            dense_scores = np.dot(self.image_embeddings, query_embedding) # [num_products]
            dense_scores = (dense_scores + 1.0) / 2.0 # Scale to [0, 1]
            
        # 3. Lexical (BM25) Search
        sparse_scores = np.zeros(len(self.catalog))
        if self.bm25 is not None:
            query_tokens = [w for w in query_text.lower().split() if len(w) > 1]
            if query_tokens:
                sparse_scores = self.bm25.get_scores(query_tokens)
                max_score = np.max(sparse_scores)
                if max_score > 0:
                    sparse_scores = sparse_scores / max_score
                    
        # 4. Hybrid Fusion
        fused_scores = (hybrid_weight * dense_scores) + ((1.0 - hybrid_weight) * sparse_scores)
        
        # Sort ONLY the filtered indices
        filtered_fused_scores = [(idx, fused_scores[idx]) for idx in filtered_indices]
        filtered_fused_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices_scores = filtered_fused_scores[:top_k]
        
        results = []
        for idx, score in top_indices_scores:
            explain_reasons = self.get_explainability(self.catalog[idx], query_text, dense_scores[idx], sparse_scores[idx])
            results.append({
                "product": self.catalog[idx],
                "dense_score": float(dense_scores[idx]),
                "sparse_score": float(sparse_scores[idx]),
                "fused_score": float(score),
                "match_reasons": explain_reasons
            })
            
        return results

    def search_by_image(self, model, query_image, top_k=6, device="cpu"):
        """
        Performs visual (image-to-image) semantic search using a query PIL Image.
        """
        if not self.catalog or self.image_embeddings is None:
            self.load()
            
        if model is None:
            raise ValueError("CLIP model is required for image search.")

        # Process image
        inputs = self.processor(images=query_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        
        # Extract query embedding
        query_embedding = model.get_image_embeddings(pixel_values, device=device)
        query_embedding = query_embedding.cpu().numpy().squeeze(0) # [dim]
        
        # Cosine similarity
        scores = np.dot(self.image_embeddings, query_embedding)
        normalized_scores = (scores + 1.0) / 2.0
        
        # Sort
        top_indices = np.argsort(normalized_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            # Explainability for image-to-image
            reasons = ["Visual Match: highly aligned shapes, colors, and textures"]
            if normalized_scores[idx] > 0.85:
                reasons.append("Exact/Close Visual Match: near-identical product outline detected")
                
            results.append({
                "product": self.catalog[idx],
                "dense_score": float(normalized_scores[idx]),
                "sparse_score": 0.0,
                "fused_score": float(normalized_scores[idx]),
                "match_reasons": reasons
            })
            
        return results

    def search_multimodal(self, model, query_image, query_text, alpha=0.5, top_k=6, device="cpu"):
        """
        Performs multimodal (Image + Text) search. Combines visual features of
        the uploaded image with modification constraints from query text.
        """
        if not self.catalog or self.image_embeddings is None:
            self.load()
            
        if model is None:
            raise ValueError("CLIP model is required for multimodal search.")

        # 1. Extract image query embedding
        inputs_img = self.processor(images=query_image, return_tensors="pt")
        pixel_values = inputs_img["pixel_values"].to(device)
        img_embedding = model.get_image_embeddings(pixel_values, device=device)
        img_embedding = img_embedding.cpu().numpy().squeeze(0) # [dim]

        # 2. Extract text query embedding
        # Parse filters out if any
        cleaned_text, min_price, max_price = self.parse_filters(query_text)
        inputs_txt = self.processor(text=[cleaned_text], return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs_txt["input_ids"].to(device)
        attention_mask = inputs_txt["attention_mask"].to(device)
        txt_embedding = model.get_text_embeddings(input_ids, attention_mask, device=device)
        txt_embedding = txt_embedding.cpu().numpy().squeeze(0) # [dim]

        # 3. Combine embeddings (weighted sum) and re-normalize
        combined_embedding = (alpha * img_embedding) + ((1.0 - alpha) * txt_embedding)
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

        # 4. Compute similarity against catalog image embeddings
        scores = np.dot(self.image_embeddings, combined_embedding)
        normalized_scores = (scores + 1.0) / 2.0

        # Apply price/brand filters if any were parsed
        filtered_indices = []
        for idx, item in enumerate(self.catalog):
            keep = True
            if min_price is not None and item["price"] < min_price:
                keep = False
            if max_price is not None and item["price"] > max_price:
                keep = False
            if keep:
                filtered_indices.append(idx)
                
        if not filtered_indices:
            filtered_indices = list(range(len(self.catalog)))

        # Sort filtered subset
        filtered_scores = [(idx, normalized_scores[idx]) for idx in filtered_indices]
        filtered_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices_scores = filtered_scores[:top_k]

        results = []
        for idx, score in top_indices_scores:
            # Multi-modal explainability
            reasons = [
                "Multimodal Fusion: combines visual shape of query image and keywords of search text",
                f"Visual Similarity: {float(np.dot(self.image_embeddings[idx], img_embedding) + 1)/2 * 100:.0f}% match to query image",
                f"Textual Alignment: {float(np.dot(self.text_embeddings[idx], txt_embedding) + 1)/2 * 100:.0f}% match to modifier text"
            ]
            if max_price is not None:
                reasons.append(f"Price Constraint: matches your price filter under ₹{max_price}")
                
            results.append({
                "product": self.catalog[idx],
                "dense_score": float(score),
                "sparse_score": 0.0,
                "fused_score": float(score),
                "match_reasons": reasons
            })

        return results

    def get_recommendations(self, product_id, top_k=4):
        """
        Finds visually and semantically similar products to a given product ID.
        Uses the cosine similarity of the item's image embedding.
        """
        if not self.catalog or self.image_embeddings is None:
            self.load()

        target_idx = -1
        for idx, item in enumerate(self.catalog):
            if item["id"] == product_id:
                target_idx = idx
                break
                
        if target_idx == -1:
            print(f"Product {product_id} not found in catalog.")
            return []
            
        target_embedding = self.image_embeddings[target_idx]
        
        # Cosine similarity
        scores = np.dot(self.image_embeddings, target_embedding)
        normalized_scores = (scores + 1.0) / 2.0
        
        # Sort excluding the target product
        sorted_indices = np.argsort(normalized_scores)[::-1]
        recommended_indices = [idx for idx in sorted_indices if idx != target_idx][:top_k]
        
        recommendations = []
        for idx in recommended_indices:
            recommendations.append({
                "product": self.catalog[idx],
                "similarity_score": float(normalized_scores[idx])
            })
            
        return recommendations
