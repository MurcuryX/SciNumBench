"""
llm_probe.py — Stage 2: LLM Hidden State Probe

Extracts table hidden-state vectors from Llama-3-8B-Instruct and
trains a Linear Probe to distinguish Clean / Fake data.

Pipeline:
  1. Table serialization (Markdown format)
  2. Forward pass, extract hidden states from layers [16, 20, 24, 28]
  3. Train Linear Probe (binary / multi-class)

Usage:
  # Feature extraction
  python llm_probe.py extract --input features/train.json --output features/train_features.npy

  # Train Probe
  python llm_probe.py train --features features/ --model models/linear_probe.pt

  # Inference
  python llm_probe.py predict --model models/linear_probe.pt --table '{"grid": [...]}'
"""

import json
import os
import argparse
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

# PyTorch imported lazily (installed on the server)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch is not installed; some features are unavailable. Run: pip install torch")

# ============================================================
# Configuration
# ============================================================

# Hidden-state extraction layers (Llama-3-8B has 32 layers)
HIDDEN_LAYERS = [16, 20, 24, 28]

# Llama-3-8B hidden-state dimension
HIDDEN_DIM = 4096

# Feature dimension = HIDDEN_DIM * len(HIDDEN_LAYERS)
FEATURE_DIM = HIDDEN_DIM * len(HIDDEN_LAYERS)

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50
PATIENCE = 5  # Early stopping: stop if val_loss does not drop for N consecutive epochs

# Label mapping (binary)
LABEL_MAP_BINARY = {
    "0_Clean": 0,
    "1_Surface": 1,
    "2_Contradiction": 1,
    "3_Camouflage": 1,
    "6_SelfContradiction": 1,
}

# Label mapping (multi-class)
LABEL_MAP_MULTI = {
    "0_Clean": 0,
    "1_Surface": 1,
    "2_Contradiction": 2,
    "3_Camouflage": 3,
    "6_SelfContradiction": 4,
}


# ============================================================
# Table serialization (Markdown format)
# ============================================================

def serialize_table_markdown(grid: List[List]) -> str:
    """
    Convert a table Grid into Markdown-formatted text.

    Example input:
      [["Variable", "Mean", "SD"], ["Score", "30", "8"], ["Age", "35", "12"]]

    Example output:
      "| Variable | Mean | SD |
       |---|---|---|
       | Score | 30 | 8 |
       | Age | 35 | 12 |"
    """
    if not grid or not grid[0]:
        return ""

    lines = []

    # Header
    header = grid[0]
    lines.append("| " + " | ".join(str(c) for c in header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    # Data rows
    for row in grid[1:]:
        # Ensure row length matches the header
        padded = list(row) + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(str(c) for c in padded[:len(header)]) + " |")

    return "\n".join(lines)


def serialize_table_natural(grid: List[List]) -> str:
    """
    Convert a table Grid into a natural-language description (fallback option).

    Example output:
      "Statistical table with 3 columns: Variable, Mean, SD.
       Row 1: Score, Mean=30, SD=8.
       Row 2: Age, Mean=35, SD=12."
    """
    if not grid or not grid[0]:
        return ""

    header = grid[0]
    lines = [f"Statistical table with {len(header)} columns: {', '.join(str(c) for c in header)}."]

    for i, row in enumerate(grid[1:], 1):
        cells = []
        for j, val in enumerate(row):
            col_name = header[j] if j < len(header) else f"col_{j}"
            cells.append(f"{col_name}={val}")
        lines.append(f"Row {i}: {', '.join(cells)}.")

    return "\n".join(lines)


# ============================================================
# LLM hidden-state extraction
# ============================================================

class HiddenStateExtractor:
    """
    Extracts hidden states from specified layers of Llama-3-8B-Instruct.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        layers: List[int] = None,
        device: str = "auto",
        max_length: int = 2048,
    ):
        """
        Args:
            model_name: HuggingFace model name or local path
            layers: which layers' hidden states to extract, default [16, 20, 24, 28]
            device: "auto", "cuda", "cpu"
            max_length: maximum token length
        """
        self.model_name = model_name
        self.layers = layers or HIDDEN_LAYERS
        self.max_length = max_length

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and tokenizer."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"[LLM] Loading model: {self.model_name}")
        print(f"[LLM] Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto" if self.device.type == "cuda" else None,
            output_hidden_states=True,  # Key: output hidden states
        )

        if self.device.type == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        print(f"[LLM] Model loaded")

    def extract_single(self, text: str) -> np.ndarray:
        """
        Extract hidden states for a single text.

        Args:
            text: serialized table text

        Returns:
            shape: (FEATURE_DIM,) = (HIDDEN_DIM * len(layers),)
        """
        if self.model is None:
            self.load_model()

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False,
        ).to(self.device)

        # Forward pass (no gradient)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # outputs.hidden_states is a tuple of length num_layers + 1
        # hidden_states[0] is the embedding-layer output
        # hidden_states[1] ~ hidden_states[32] are the outputs of layers 1~32
        hidden_states = outputs.hidden_states

        # Extract the specified layers' hidden states, taking the last token's vector
        features = []
        for layer_idx in self.layers:
            # layer_idx starts at 1, so use hidden_states[layer_idx]
            layer_hidden = hidden_states[layer_idx]  # shape: (1, seq_len, hidden_dim)
            last_token = layer_hidden[0, -1, :]  # shape: (hidden_dim,)
            features.append(last_token.cpu().float().numpy())

        # Concatenate features from all layers
        return np.concatenate(features)  # shape: (FEATURE_DIM,)

    def extract_batch(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Extract hidden states in batches.

        Args:
            texts: list of texts
            batch_size: batch size

        Returns:
            shape: (len(texts), FEATURE_DIM)
        """
        if self.model is None:
            self.load_model()

        all_features = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting hidden states"):
            batch_texts = texts[i:i + batch_size]

            # Tokenize (with padding)
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            hidden_states = outputs.hidden_states

            # Extract the last non-padding token's hidden state for each sample
            for j in range(len(batch_texts)):
                features = []
                for layer_idx in self.layers:
                    layer_hidden = hidden_states[layer_idx]
                    # Find the position of the last non-padding token
                    attention_mask = inputs["attention_mask"][j]
                    last_token_idx = attention_mask.sum().item() - 1
                    last_token = layer_hidden[j, last_token_idx, :]
                    features.append(last_token.cpu().float().numpy())
                all_features.append(np.concatenate(features))

        return np.array(all_features)


# ============================================================
# Dataset (requires PyTorch)
# ============================================================

if TORCH_AVAILABLE:
    class TableFeatureDataset(Dataset):
        """Table feature dataset."""

        def __init__(self, features: np.ndarray, labels: np.ndarray):
            self.features = torch.FloatTensor(features)
            self.labels = torch.LongTensor(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]


# ============================================================
# Linear Probe model (requires PyTorch)
# ============================================================

if TORCH_AVAILABLE:
    class LinearProbe(nn.Module):
        """
        Linear probe: single-layer logistic regression.
        Used to detect anomalies from LLM hidden states.
        """

        def __init__(self, input_dim: int = FEATURE_DIM, num_classes: int = 2):
            super().__init__()
            self.linear = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            return self.linear(x)


    class MLPProbe(nn.Module):
        """
        Two-layer MLP probe: stronger than the linear probe.
        """

        def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 256, num_classes: int = 2):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):
            return self.network(x)


# ============================================================
# Trainer (requires PyTorch)
# ============================================================

if TORCH_AVAILABLE:
    class ProbeTrainer:
        """Probe trainer."""

        def __init__(
            self,
            model: nn.Module,
            device: str = "auto",
            learning_rate: float = LEARNING_RATE,
        ):
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            self.model = model.to(self.device)
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            self.criterion = nn.CrossEntropyLoss()

        def train_epoch(self, dataloader: DataLoader) -> float:
            """Train one epoch, return the average loss."""
            self.model.train()
            total_loss = 0.0

            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            return total_loss / len(dataloader)

        @torch.no_grad()
        def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
            """Evaluate the model, return metrics."""
            self.model.eval()
            total_loss = 0.0
            all_preds = []
            all_labels = []

            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            # Compute metrics
            accuracy = (all_preds == all_labels).mean()

            # Binary-classification metrics
            if len(np.unique(all_labels)) == 2:
                tp = ((all_preds == 1) & (all_labels == 1)).sum()
                fp = ((all_preds == 1) & (all_labels == 0)).sum()
                fn = ((all_preds == 0) & (all_labels == 1)).sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision = recall = f1 = 0.0

            return {
                "loss": total_loss / len(dataloader),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        def train(
            self,
            train_features: np.ndarray,
            train_labels: np.ndarray,
            val_features: np.ndarray,
            val_labels: np.ndarray,
            epochs: int = EPOCHS,
            batch_size: int = BATCH_SIZE,
            patience: int = PATIENCE,
            save_path: str = None,
        ) -> Dict[str, List[float]]:
            """
            Full training loop.

            Returns:
                Training history {"train_loss": [...], "val_loss": [...], ...}
            """
            # Create datasets
            train_dataset = TableFeatureDataset(train_features, train_labels)
            val_dataset = TableFeatureDataset(val_features, val_labels)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Training history
            history = {
                "train_loss": [],
                "val_loss": [],
                "val_accuracy": [],
                "val_f1": [],
            }

            best_val_loss = float("inf")
            patience_counter = 0

            print(f"[TRAIN] Starting training | Epochs={epochs} | Batch={batch_size}")
            print(f"[TRAIN] Train set: {len(train_dataset)} | Val set: {len(val_dataset)}")

            for epoch in range(epochs):
                # Train
                train_loss = self.train_epoch(train_loader)

                # Validate
                val_metrics = self.evaluate(val_loader)

                # Record history
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                history["val_f1"].append(val_metrics["f1"])

                # Print progress
                print(f"  Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Val Acc: {val_metrics['accuracy']:.4f} | "
                      f"Val F1: {val_metrics['f1']:.4f}")

                # Early-stopping check
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0

                    # Save the best model
                    if save_path:
                        torch.save({
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "epoch": epoch,
                            "val_loss": val_metrics["loss"],
                            "val_metrics": val_metrics,
                        }, save_path)
                        print(f"  -> Saved best model to {save_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  -> Early stopping: val_loss did not drop for {patience} consecutive epochs")
                        break

            return history


# ============================================================
# Predictor (requires PyTorch)
# ============================================================

if TORCH_AVAILABLE:
    class ProbePredictor:
        """Probe predictor."""

        def __init__(
            self,
            model_path: str,
            model_type: str = "linear",  # "linear" or "mlp"
            num_classes: int = 2,
            device: str = "auto",
        ):
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            # Create model
            if model_type == "linear":
                self.model = LinearProbe(num_classes=num_classes)
            else:
                self.model = MLPProbe(num_classes=num_classes)

            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            # Label mapping
            if num_classes == 2:
                self.label_names = ["Clean", "Fake"]
            else:
                self.label_names = ["Clean", "L1", "L2", "L3", "L6"]

        @torch.no_grad()
        def predict(self, feature: np.ndarray) -> Dict[str, Any]:
            """
            Predict a single sample.

            Args:
                feature: shape (FEATURE_DIM,)

            Returns:
                {"label": "Clean", "confidence": 0.95, "probs": [0.95, 0.05]}
            """
            x = torch.FloatTensor(feature).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]

            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()

            return {
                "label": self.label_names[pred_idx],
                "confidence": confidence,
                "probs": {name: probs[i].item() for i, name in enumerate(self.label_names)},
            }

        @torch.no_grad()
        def predict_batch(self, features: np.ndarray) -> List[Dict[str, Any]]:
            """Batch prediction."""
            x = torch.FloatTensor(features).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)

            results = []
            for i in range(len(features)):
                pred_idx = probs[i].argmax().item()
                confidence = probs[i][pred_idx].item()
                results.append({
                    "label": self.label_names[pred_idx],
                    "confidence": confidence,
                    "probs": {name: probs[i][j].item() for j, name in enumerate(self.label_names)},
                })

            return results


# ============================================================
# Full pipeline: from database to prediction
# ============================================================

def run_extraction_pipeline(
    db_path: str,
    output_path: str,
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    batch_size: int = 8,
):
    """
    From database: extract tables -> serialize -> extract hidden states -> save.

    Args:
        db_path: path to arxiv_data.db
        output_path: output .npy file path
        model_name: LLM model name
        batch_size: batch size
    """
    import sqlite3

    # 1. Load scinum_bench from the database
    print("[PIPELINE] Loading scinum_bench data...")
    conn = sqlite3.connect(db_path)

    rows = conn.execute("""
        SELECT bench_id, arxiv_id, dataset_split, original_grid, corrupted_grid,
               corruption_level
        FROM scinum_bench
    """).fetchall()

    conn.close()
    print(f"[PIPELINE] Loaded {len(rows)} records")

    # 2. Serialize
    print("[PIPELINE] Serializing tables...")
    texts = []
    labels = []
    bench_ids = []

    for bench_id, arxiv_id, split, orig_grid, corr_grid, level in tqdm(rows, desc="Serializing"):
        # Use corrupted_grid (Fake samples) or original_grid (Clean samples)
        grid = json.loads(corr_grid)
        text = serialize_table_markdown(grid)
        texts.append(text)

        # Label
        label = LABEL_MAP_BINARY.get(level, 0)
        labels.append(label)
        bench_ids.append(bench_id)

    labels = np.array(labels)
    bench_ids = np.array(bench_ids)

    # 3. Extract hidden states
    print("[PIPELINE] Extracting hidden states...")
    extractor = HiddenStateExtractor(model_name=model_name)
    extractor.load_model()
    features = extractor.extract_batch(texts, batch_size=batch_size)

    # 4. Save
    print(f"[PIPELINE] Saving features to {output_path}")
    np.save(output_path, features)

    # Save labels and IDs
    label_path = output_path.replace(".npy", "_labels.npy")
    id_path = output_path.replace(".npy", "_ids.npy")
    np.save(label_path, labels)
    np.save(id_path, bench_ids)

    print(f"[PIPELINE] Done. Feature shape: {features.shape}")
    return features, labels, bench_ids


def run_training_pipeline(
    features_dir: str,
    output_model: str,
    model_type: str = "linear",
    num_classes: int = 2,
):
    """
    Train the Probe model.

    Args:
        features_dir: feature directory (contains train_features.npy, val_features.npy, etc.)
        output_model: output model path
        model_type: "linear" or "mlp"
        num_classes: number of classes (2 or 5)
    """
    # 1. Load features
    print("[TRAIN] Loading features...")
    train_features = np.load(os.path.join(features_dir, "train_features.npy"))
    train_labels = np.load(os.path.join(features_dir, "train_labels.npy"))
    val_features = np.load(os.path.join(features_dir, "val_features.npy"))
    val_labels = np.load(os.path.join(features_dir, "val_labels.npy"))

    print(f"[TRAIN] Train set: {train_features.shape[0]} samples, {train_features.shape[1]} dims")
    print(f"[TRAIN] Val set: {val_features.shape[0]} samples")

    # 2. Create model
    if model_type == "linear":
        model = LinearProbe(input_dim=train_features.shape[1], num_classes=num_classes)
    else:
        model = MLPProbe(input_dim=train_features.shape[1], num_classes=num_classes)

    print(f"[TRAIN] Model type: {model_type}")
    print(f"[TRAIN] Parameter count: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Train
    trainer = ProbeTrainer(model)
    history = trainer.train(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        save_path=output_model,
    )

    # 4. Save training history
    history_path = output_model.replace(".pt", "_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"[TRAIN] Training done. Best model: {output_model}")
    return history


# ============================================================
# CLI entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LLM Hidden State Probe")
    subparsers = parser.add_subparsers(dest="command", help="subcommands")

    # extract subcommand
    extract_parser = subparsers.add_parser("extract", help="Extract hidden-state features")
    extract_parser.add_argument("--db", type=str, default="arxiv_data.db", help="database path")
    extract_parser.add_argument("--output", type=str, required=True, help="output .npy path")
    extract_parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                                help="LLM model name or path")
    extract_parser.add_argument("--batch-size", type=int, default=8, help="batch size")

    # train subcommand
    train_parser = subparsers.add_parser("train", help="Train the Probe")
    train_parser.add_argument("--features", type=str, required=True, help="feature directory")
    train_parser.add_argument("--output", type=str, required=True, help="output model path")
    train_parser.add_argument("--model-type", type=str, default="linear",
                              choices=["linear", "mlp"], help="model type")
    train_parser.add_argument("--num-classes", type=int, default=2, help="number of classes")

    # predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Predict")
    predict_parser.add_argument("--model", type=str, required=True, help="model path")
    predict_parser.add_argument("--model-type", type=str, default="linear",
                                choices=["linear", "mlp"], help="model type")
    predict_parser.add_argument("--table", type=str, required=True, help="table JSON")

    args = parser.parse_args()

    if args.command == "extract":
        run_extraction_pipeline(
            db_path=args.db,
            output_path=args.output,
            model_name=args.model,
            batch_size=args.batch_size,
        )
    elif args.command == "train":
        run_training_pipeline(
            features_dir=args.features,
            output_model=args.output,
            model_type=args.model_type,
            num_classes=args.num_classes,
        )
    elif args.command == "predict":
        grid = json.loads(args.table)
        text = serialize_table_markdown(grid)

        predictor = ProbePredictor(
            model_path=args.model,
            model_type=args.model_type,
        )

        # Features must be extracted first, which requires the LLM model.
        # In practice, extract features first and then predict.
        print("[PREDICT] Features must be extracted first; please use the full pipeline")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
