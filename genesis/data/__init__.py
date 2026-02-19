"""Data handling for Genesis."""

__all__ = [
    "DatasetLoader",
    "PubMedQADataset",
    "TTSDataset",
    "TextPreprocessor",
    "AudioPreprocessor",
]


def __getattr__(name):
    if name in ("DatasetLoader", "PubMedQADataset", "TTSDataset"):
        from genesis.data.datasets import DatasetLoader, PubMedQADataset, TTSDataset
        return {"DatasetLoader": DatasetLoader, "PubMedQADataset": PubMedQADataset,
                "TTSDataset": TTSDataset}[name]
    if name in ("TextPreprocessor", "AudioPreprocessor"):
        from genesis.data.preprocessing import TextPreprocessor, AudioPreprocessor
        return {"TextPreprocessor": TextPreprocessor,
                "AudioPreprocessor": AudioPreprocessor}[name]
    raise AttributeError(f"module 'genesis.data' has no attribute {name!r}")
