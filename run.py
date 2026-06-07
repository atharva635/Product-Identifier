import os
import sys
import argparse
# pyrefly: ignore [missing-import]


# --- Virtual Environment Self-Elevation ---
# If this script is run with the system Python, automatically restart it using the virtual environment Python.
def elevate_to_venv():
    if os.name == "nt":
        venv_python = os.path.abspath(os.path.join(".venv", "Scripts", "python.exe"))
    else:
        venv_python = os.path.abspath(os.path.join(".venv", "bin", "python"))
        
    if os.path.exists(venv_python):
        current_exe = os.path.abspath(sys.executable)
        if current_exe != venv_python:
            print(f"[System] Re-executing script within virtual environment: {venv_python}")
            # Spawn the process in venv and replace current process
            os.execv(venv_python, [venv_python] + sys.argv)
    else:
        print("[Warning] Virtual environment (.venv) not found. Running with current python interpreter.")

# Run elevation before importing ML libraries (which might not be in the global environment)
elevate_to_venv()

# Now we can safely import project modules
import uvicorn
from src.model.dataset import generate_synthetic_catalog
from src.search.indexer import build_vector_index

def main():
    parser = argparse.ArgumentParser(description="OmniSearch-ML Orchestrator")
    parser.add_argument("--prep", action="store_true", help="Generate synthetic product catalog")
    parser.add_argument("--train", action="store_true", help="Run CLIP fine-tuning locally")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--index", action="store_true", help="Rebuild vector search index embeddings")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    
    args = parser.parse_args()
    
    catalog_path = "data/catalog.json"
    embeddings_path = "data/embeddings.npz"
    checkpoint_path = "data/checkpoints/best_model.pt"

    # Stage 1: Data Preparation
    # If explicit prep requested, or catalog doesn't exist, generate it
    if args.prep or not os.path.exists(catalog_path):
        print("\n=== Stage 1: Generating Product Catalog & Images ===")
        generate_synthetic_catalog(output_dir="data", count_per_template=8) # Creates 88 items

    # Stage 2: Training (Optional)
    if args.train:
        print(f"\n=== Stage 2: Fine-Tuning CLIP (Epochs: {args.epochs}) ===")
        from src.model.train import train_clip
        train_clip(epochs=args.epochs, catalog_path=catalog_path, output_dir="data")

    # Stage 3: Indexing
    # Index if requested, or if embeddings file doesn't exist
    if args.index or not os.path.exists(embeddings_path):
        print("\n=== Stage 3: Pre-generating Vector Search Embeddings ===")
        build_vector_index(
            catalog_path=catalog_path,
            checkpoint_path=checkpoint_path,
            output_path=embeddings_path
        )

    # Stage 4: API Serving
    print(f"\n=== Stage 4: Starting API Server on http://localhost:{args.port} ===")
    print("Press Ctrl+C to stop the server.")
    
    # Run FastAPI via Uvicorn
    # We load app via string import to support hot-reloading if desired, though standard run is fine
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=args.port, reload=False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Shutdown] Server stopped by user.")
        sys.exit(0)
