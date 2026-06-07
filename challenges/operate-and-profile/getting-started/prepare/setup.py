"""
Download the DeepSeek-V2-Lite checkpoint, convert to DCP, and tokenize the DCLM corpus.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

if __name__ == "__main__":
    kwargs = dict()
    kwargs["repo_id"] = "deepseek-ai/DeepSeek-V2-Lite"
    kwargs["local_dir"] = "checkpoints/deepseek-v2-lite/hf-import"
    snapshot_download(**kwargs)

if __name__ == "__main__":
    kwargs = dict()
    kwargs["repo_id"] = "mlfoundations/dclm-baseline-1.0"
    kwargs["repo_type"] = "dataset"
    kwargs["local_dir"] = "datasets/dclm-baseline/rawtxt"
    kwargs["allow_patterns"] = "global-shard_03_of_10/local-shard_1_of_10/shard_0000000[0-3]_processed.jsonl.zst"
    snapshot_download(**kwargs)

if __name__ == "__main__":
    from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg, launch

    cfg = ConvertCheckpointCfg()
    cfg.operation = "hf2dcp"
    cfg.load_path = Path("checkpoints/deepseek-v2-lite/hf-import")
    cfg.save_path = Path("checkpoints/deepseek-v2-lite/torch-dcp/step-00000000")
    if not (cfg.save_path / ".metadata").exists():
        launch(cfg)

if __name__ == "__main__":
    from pithtrain.tasks.build_tokenized_corpus import BuildTokenizedCorpusCfg, launch

    cfg = BuildTokenizedCorpusCfg()
    cfg.tokenizer_name = "checkpoints/deepseek-v2-lite/hf-import"
    cfg.source_path = Path("datasets/dclm-baseline/rawtxt")
    cfg.output_path = Path("datasets/dclm-baseline/toktxt/deepseek-v2")
    launch(cfg)
