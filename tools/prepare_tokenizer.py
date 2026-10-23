"""Build tokenizer.mtok for MNN 3.6 from the upstream SD1.5 vocab/merges pair."""
import argparse
import sys
from pathlib import Path


def prepare(model: Path, mnn_root: Path) -> Path:
    from transformers import CLIPTokenizerFast

    tokenizer = CLIPTokenizerFast(
        vocab_file=str(model / "vocab.json"),
        merges_file=str(model / "merges.txt"),
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    tokenizer.save_pretrained(str(model))
    export_root = mnn_root / "transformers" / "llm" / "export"
    sys.path.insert(0, str(export_root))
    from utils.tokenizer import LlmTokenizer

    exported = Path(LlmTokenizer(str(model), "clip").export(str(model), model_path=str(model), model_type="clip"))
    if exported.name != "tokenizer.mtok" or not exported.is_file():
        raise RuntimeError("MNN tokenizer export failed")
    expected = tokenizer("a cute cat", add_special_tokens=True).input_ids
    print(f"tokenizer: {exported} ({exported.stat().st_size} bytes)")
    print(f"reference tokens for 'a cute cat': {expected}")
    return exported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("mnn_root", type=Path)
    args = parser.parse_args()
    prepare(args.model, args.mnn_root)
