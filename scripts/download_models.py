#!/usr/bin/env python3
"""
Model Download Helper for Genesis

Downloads and caches popular models for experiments.
"""

import argparse
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Available models for download
AVAILABLE_MODELS = {
    # LLM Models
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "phi-2": "microsoft/phi-2",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",

    # TTS Models (placeholders)
    "tacotron2": "nvidia/tacotron2",
}


def download_model(model_name: str, cache_dir: str = None) -> bool:
    """
    Download a model from HuggingFace.

    Args:
        model_name: Model name or key from AVAILABLE_MODELS
        cache_dir: Directory to cache model

    Returns:
        True if successful, False otherwise
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers")
        return False

    # Resolve model name
    if model_name in AVAILABLE_MODELS:
        model_path = AVAILABLE_MODELS[model_name]
    else:
        model_path = model_name

    logger.info(f"Downloading model: {model_path}")

    try:
        # Download tokenizer
        logger.info("  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        # Download model
        logger.info("  Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )

        logger.info(f"  Model downloaded successfully!")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Clean up
        del model
        del tokenizer

        return True

    except Exception as e:
        logger.error(f"  Failed to download model: {e}")
        return False


def download_dataset(dataset_name: str, cache_dir: str = None) -> bool:
    """
    Download a dataset from HuggingFace.

    Args:
        dataset_name: Dataset name
        cache_dir: Directory to cache dataset

    Returns:
        True if successful, False otherwise
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed. Run: pip install datasets")
        return False

    logger.info(f"Downloading dataset: {dataset_name}")

    try:
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        logger.info(f"  Dataset downloaded successfully!")
        logger.info(f"  Splits: {list(dataset.keys())}")
        return True

    except Exception as e:
        logger.error(f"  Failed to download dataset: {e}")
        return False


def list_available_models():
    """Print available models."""
    print("\nAvailable Models:")
    print("-" * 50)
    for key, path in AVAILABLE_MODELS.items():
        print(f"  {key:20} -> {path}")
    print()


def check_huggingface_token():
    """Check if HuggingFace token is configured."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if token:
        logger.info("HuggingFace token found in environment")
        return True

    # Check for token file
    token_file = Path.home() / ".huggingface" / "token"
    if token_file.exists():
        logger.info("HuggingFace token found in ~/.huggingface/token")
        return True

    logger.warning("No HuggingFace token found.")
    logger.warning("Some models (like Llama) require authentication.")
    logger.warning("Run: huggingface-cli login")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download models for Genesis experiments"
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="Model names to download (use --list to see available)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for downloads",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        action="append",
        help="Dataset to download (can be specified multiple times)",
    )
    parser.add_argument(
        "--all-small",
        action="store_true",
        help="Download all small models (gpt2, tinyllama, phi-2)",
    )
    args = parser.parse_args()

    if args.list:
        list_available_models()
        return

    # Check token
    check_huggingface_token()

    # Collect models to download
    models_to_download = list(args.models or [])

    if args.all_small:
        models_to_download.extend(["gpt2", "tinyllama", "phi-2"])

    if not models_to_download and not args.dataset:
        parser.print_help()
        print("\nExamples:")
        print("  python download_models.py gpt2")
        print("  python download_models.py --all-small")
        print("  python download_models.py --dataset pubmed_qa")
        print("  python download_models.py --list")
        return

    # Download models
    success_count = 0
    fail_count = 0

    for model_name in models_to_download:
        if download_model(model_name, args.cache_dir):
            success_count += 1
        else:
            fail_count += 1

    # Download datasets
    if args.dataset:
        for dataset_name in args.dataset:
            if download_dataset(dataset_name, args.cache_dir):
                success_count += 1
            else:
                fail_count += 1

    # Summary
    print()
    print("=" * 50)
    print(f"Downloads complete: {success_count} succeeded, {fail_count} failed")
    print("=" * 50)


if __name__ == "__main__":
    main()
