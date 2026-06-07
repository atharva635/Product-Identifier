import os
import json
import time
import threading
import traceback
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch
import psutil

from src.model.model import ECommerceCLIP
from src.model.train import train_clip
from src.search.engine import HybridSearchEngine
from src.search.indexer import build_vector_index

app = FastAPI(title="OmniSearch-ML API", description="Advanced Multimodal Product Retrieval System")

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model, engine, training state, and latency metrics
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = None
SEARCH_ENGINE = None
SEARCH_LATENCIES = []

TRAINING_STATE = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "status": "idle",
    "error": None
}

# Mount data images directory if it exists
os.makedirs("data/images", exist_ok=True)
app.mount("/data/images", StaticFiles(directory="data/images"), name="images")

# Pydantic Schemas
class SearchQuery(BaseModel):
    query: str
    weight: float = 0.5
    top_k: int = 6

class ChatRequest(BaseModel):
    message: str

# Helper: Load model and search engine
def load_models():
    global CLIP_MODEL, SEARCH_ENGINE
    
    catalog_path = "data/catalog.json"
    checkpoint_path = "data/checkpoints/best_model.pt"
    embeddings_path = "data/embeddings.npz"
    
    # Initialize CLIP model
    if CLIP_MODEL is None:
        CLIP_MODEL = ECommerceCLIP()
        if os.path.exists(checkpoint_path):
            print(f"Loading custom checkpoint: {checkpoint_path}")
            CLIP_MODEL.load_checkpoint(checkpoint_path, device=DEVICE)
        CLIP_MODEL.to(DEVICE)
        CLIP_MODEL.eval()
        
    # Initialize Search Engine
    if SEARCH_ENGINE is None:
        SEARCH_ENGINE = HybridSearchEngine(
            catalog_path=catalog_path,
            embeddings_path=embeddings_path
        )
        if os.path.exists(catalog_path):
            SEARCH_ENGINE.load()
        else:
            print("Catalog not found. Engine is uninitialized.")

def record_latency(start_time):
    duration_ms = (time.time() - start_time) * 1000
    SEARCH_LATENCIES.append(duration_ms)
    if len(SEARCH_LATENCIES) > 50:
        SEARCH_LATENCIES.pop(0)

@app.on_event("startup")
async def startup_event():
    # Load model and indexes on startup if possible
    try:
        load_models()
    except Exception as e:
        print(f"Startup warning: Models could not be loaded yet. {e}")

# Background worker for training
def train_and_index_worker(epochs: int):
    global CLIP_MODEL, SEARCH_ENGINE, TRAINING_STATE
    TRAINING_STATE["is_training"] = True
    TRAINING_STATE["status"] = "Training model (InfoNCE contrastive learning)..."
    TRAINING_STATE["error"] = None
    TRAINING_STATE["total_epochs"] = epochs
    
    try:
        # 1. Run Fine-Tuning
        train_clip(epochs=epochs, catalog_path="data/catalog.json", output_dir="data")
        
        # 2. Re-build the vector index with the newly trained model
        TRAINING_STATE["status"] = "Rebuilding vector index..."
        build_vector_index(
            catalog_path="data/catalog.json",
            checkpoint_path="data/checkpoints/best_model.pt",
            output_path="data/embeddings.npz",
            device=DEVICE
        )
        
        # 3. Reload Search Engine and Model
        TRAINING_STATE["status"] = "Reloading models into memory..."
        CLIP_MODEL = None
        SEARCH_ENGINE = None
        load_models()
        
        TRAINING_STATE["status"] = "completed"
        print("Background training and re-indexing complete.")
        
    except Exception as e:
        print(f"Error in training worker: {e}")
        traceback.print_exc()
        TRAINING_STATE["status"] = "failed"
        TRAINING_STATE["error"] = str(e)
    finally:
        TRAINING_STATE["is_training"] = False

# API Routes

@app.get("/api/catalog")
def get_catalog():
    """Returns the complete product catalog metadata."""
    catalog_path = "data/catalog.json"
    if not os.path.exists(catalog_path):
        return []
    with open(catalog_path, "r") as f:
        return json.load(f)

@app.post("/api/search/text")
def search_text(query_data: SearchQuery):
    """Hybrid (Lexical BM25 + Semantic Vector) text search with price filter support."""
    global CLIP_MODEL, SEARCH_ENGINE
    start_time = time.time()
    
    if SEARCH_ENGINE is None or not SEARCH_ENGINE.catalog:
        load_models()
        
    if not SEARCH_ENGINE.catalog:
        raise HTTPException(status_code=400, detail="Catalog has not been built. Please seed data first.")
        
    try:
        results = SEARCH_ENGINE.search_by_text(
            model=CLIP_MODEL,
            query_text=query_data.query,
            hybrid_weight=query_data.weight,
            top_k=query_data.top_k,
            device=DEVICE
        )
        record_latency(start_time)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/search/image")
async def search_image(
    file: UploadFile = File(...),
    top_k: int = Form(6)
):
    """Pure visual (image-to-image) semantic search via uploaded image."""
    global CLIP_MODEL, SEARCH_ENGINE
    start_time = time.time()
    
    if SEARCH_ENGINE is None or SEARCH_ENGINE.image_embeddings is None:
        load_models()
        
    if SEARCH_ENGINE.image_embeddings is None:
        raise HTTPException(status_code=400, detail="Vector index does not exist. Please run indexer first.")
        
    try:
        contents = await file.read()
        import io
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        results = SEARCH_ENGINE.search_by_image(
            model=CLIP_MODEL,
            query_image=img,
            top_k=top_k,
            device=DEVICE
        )
        record_latency(start_time)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visual search failed: {str(e)}")

@app.post("/api/search/multimodal")
async def search_multimodal(
    file: UploadFile = File(...),
    query: str = Form(""),
    alpha: float = Form(0.5),
    top_k: int = Form(6)
):
    """Multimodal retrieval combining query image visual features and text modifiers."""
    global CLIP_MODEL, SEARCH_ENGINE
    start_time = time.time()

    if SEARCH_ENGINE is None or SEARCH_ENGINE.image_embeddings is None:
        load_models()

    if SEARCH_ENGINE.image_embeddings is None:
        raise HTTPException(status_code=400, detail="Vector index does not exist. Please run indexer first.")

    try:
        contents = await file.read()
        import io
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        results = SEARCH_ENGINE.search_multimodal(
            model=CLIP_MODEL,
            query_image=img,
            query_text=query,
            alpha=alpha,
            top_k=top_k,
            device=DEVICE
        )
        record_latency(start_time)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal search failed: {str(e)}")

@app.get("/api/recommendations/{product_id}")
def get_recommendations(product_id: str, top_k: int = 4):
    """Visual content-based recommendation for a product ID."""
    global SEARCH_ENGINE
    
    if SEARCH_ENGINE is None or not SEARCH_ENGINE.catalog:
        load_models()
        
    if not SEARCH_ENGINE.catalog:
         raise HTTPException(status_code=400, detail="Catalog metadata is not available.")
         
    try:
        recommendations = SEARCH_ENGINE.get_recommendations(product_id, top_k=top_k)
        return {"status": "success", "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.get("/api/embeddings/viz")
def get_embeddings_viz(method: str = "pca"):
    """Applies PCA or t-SNE reduction on CLIP image embeddings for 2D visualization."""
    global SEARCH_ENGINE
    if SEARCH_ENGINE is None or SEARCH_ENGINE.image_embeddings is None:
        load_models()

    if SEARCH_ENGINE.image_embeddings is None:
        return []

    try:
        embeddings = SEARCH_ENGINE.image_embeddings
        catalog = SEARCH_ENGINE.catalog

        if method.lower() == "tsne":
            from sklearn.manifold import TSNE
            # With small catalogs, perplexity needs to be small
            perplexity = min(5, len(catalog) - 1)
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=250)
        else:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)

        coords_2d = reducer.fit_transform(embeddings)

        viz_data = []
        for idx, item in enumerate(catalog):
            viz_data.append({
                "id": item["id"],
                "title": item["title"],
                "brand": item["brand"],
                "category": item["category"],
                "price": item["price"],
                "image_path": item["image_path"],
                "x": float(coords_2d[idx, 0]),
                "y": float(coords_2d[idx, 1])
            })
        return viz_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dimensionality reduction failed: {str(e)}")

@app.post("/api/chat")
def chat_assistant(request: ChatRequest):
    """Conversational RAG chatbot suggesting products based on hybrid search queries."""
    global CLIP_MODEL, SEARCH_ENGINE
    if SEARCH_ENGINE is None or not SEARCH_ENGINE.catalog:
        load_models()

    if not SEARCH_ENGINE.catalog:
        return {"response": "The product catalog is currently uninitialized. Please build index first!"}

    try:
        # Search top 3 items for chatbot query
        results = SEARCH_ENGINE.search_by_text(
            model=CLIP_MODEL,
            query_text=request.message,
            hybrid_weight=0.7,
            top_k=3,
            device=DEVICE
        )

        if not results:
            return {
                "response": "I searched our store but couldn't find items that match your description. Can you specify a different brand, category, or style?",
                "retrieved_products": []
            }

        # Build natural conversational response
        intro = f"Hi! I searched our product catalog and found these top recommendations for you:\n\n"
        items_desc = []
        for idx, res in enumerate(results):
            prod = res["product"]
            match_pct = int(res["fused_score"] * 100)
            reasons = " • ".join(res["match_reasons"][:2])
            desc = (f"🛍️ **{prod['title']}** (₹{prod['price']:,})\n"
                    f"   *Category: {prod['category']} by {prod['brand']}* | **{match_pct}% Match**\n"
                    f"   _{prod['description']}_\n"
                    f"   *Why this matches:* {reasons}\n")
            items_desc.append(desc)

        outro = "\nFeel free to ask me to filter by price, search for different colors, or ask for another category!"
        response_text = intro + "\n".join(items_desc) + outro

        return {
            "response": response_text,
            "retrieved_products": [r["product"] for r in results]
        }
    except Exception as e:
        return {"response": f"Oops! I had trouble looking that up: {str(e)}", "retrieved_products": []}

@app.get("/api/metrics")
def get_metrics():
    """Returns training details, latency metrics, and hardware resource stats."""
    catalog_path = "data/catalog.json"
    history_path = "data/history.json"
    
    # Catalog Stats
    num_products = 0
    categories = {}
    if os.path.exists(catalog_path):
        with open(catalog_path, "r") as f:
            catalog = json.load(f)
            num_products = len(catalog)
            for item in catalog:
                categories[item["category"]] = categories.get(item["category"], 0) + 1
                
    # Training History
    history = {}
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
            
    checkpoint_exists = os.path.exists("data/checkpoints/best_model.pt")
    
    # Latency Stats
    avg_latency = float(np.mean(SEARCH_LATENCIES)) if SEARCH_LATENCIES else 0.0
    p95_latency = float(np.percentile(SEARCH_LATENCIES, 95)) if SEARCH_LATENCIES else 0.0
    
    # Resource Monitor (psutil)
    cpu_usage = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    ram_usage = memory_info.percent
    
    gpu_name = "CPU Only"
    gpu_mem_allocated = 0.0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2) # MB
        
    return {
        "device": DEVICE,
        "checkpoint_active": checkpoint_exists,
        "num_products": num_products,
        "categories": categories,
        "history": history,
        "training_state": TRAINING_STATE,
        "performance": {
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "cpu_util_pct": cpu_usage,
            "ram_util_pct": ram_usage,
            "gpu_hardware": gpu_name,
            "gpu_mem_allocated_mb": gpu_mem_allocated
        }
    }

@app.post("/api/train")
def run_training(epochs: int = Form(3), background_tasks: BackgroundTasks = BackgroundTasks()):
    """Triggers background fine-tuning of CLIP and index rebuild."""
    global TRAINING_STATE
    
    if TRAINING_STATE["is_training"]:
        return {"status": "error", "message": "Training is already in progress."}
        
    background_tasks.add_task(train_and_index_worker, epochs)
    return {"status": "success", "message": f"Training initiated for {epochs} epochs in the background."}

# Mount frontend files
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
