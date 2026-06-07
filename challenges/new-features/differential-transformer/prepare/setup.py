"""
Download the DeepSeek-V2-Lite tokenizer and the DCLM dataset.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

if __name__ == "__main__":
    kwargs = dict()
    kwargs["repo_id"] = "mlfoundations/dclm-baseline-1.0"
    kwargs["local_dir"] = "datasets/dclm-baseline/rawtxt"
    pattern = "global-shard_03_of_10/local-shard_1_of_10/shard_0000000[0-3]_processed.jsonl.zst"
    kwargs["allow_patterns"] = [pattern]
    snapshot_download(**kwargs, repo_type="dataset")

if __name__ == "__main__":
    kwargs = dict()
    kwargs["repo_id"] = "deepseek-ai/DeepSeek-V2-Lite"
    kwargs["local_dir"] = "checkpoints/deepseek-v2-lite"
    pattern = "*tokenizer*"
    kwargs["allow_patterns"] = [pattern]
    snapshot_download(**kwargs, repo_type="model")

if __name__ == "__main__":
    from pithtrain.tasks.build_tokenized_corpus import BuildTokenizedCorpusCfg, launch

    cfg = BuildTokenizedCorpusCfg()
    cfg.tokenizer_name = Path("checkpoints/deepseek-v2-lite")
    cfg.source_path = Path("datasets/dclm-baseline/rawtxt")
    cfg.output_path = Path("datasets/dclm-baseline/toktxt")

    launch(cfg)
