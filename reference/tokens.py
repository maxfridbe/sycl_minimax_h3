"""tokens.py - the reference's token ids for prompts, for checking the Rust tokenizer.

    python3 tokens.py <out.json> <prompt file> [more prompt files ...]

Writes {"<name>": {"text": ..., "ids": [...]}, ...} with the MiniMax H3 tokenizer exactly as h3x.py uses it (the
prompt read and stripped as `--prompt-file` does), plus a set of awkward strings (numbers, punctuation runs,
newlines, non-Latin scripts, the extra special tokens) under "case.N".
"""
import json
import sys

sys.path.insert(0, "/comfy")
_argv, sys.argv = sys.argv, [sys.argv[0], "--cpu"]   # tokenizing needs no GPU: ComfyUI's CPU mode
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
from comfy.text_encoders.minimax import MiniMaxH3Tokenizer  # noqa: E402
sys.argv = _argv

CASES = [
    "Hello world", "  leading spaces", "trailing spaces   ", "tabs\tand\nnewlines\n\n\nend", "12345678 3.14159 1,000,000",
    "don't won't I'm you're they've we'll he'd", "DON'T SHOUT", "!!!??? ... --- ***", "émigré naïve café", "日本語のテキスト",
    "Привет, мир", "مرحبا بالعالم", "emoji 🎬🎥 here", "<d>tagged</d> <|caption_start|>cap<|caption_end|>", "x" * 50,
    "a\r\nb", "   ", "line one.\nLine two!\n", "<Picture 1>: ",
]


def ids(tok, text):
    return [int(t[0]) for t in tok.tokenize_with_weights(text)["qwen3vl_32b"][0]]


def main():
    tok = MiniMaxH3Tokenizer()
    out = {}
    for path in sys.argv[2:]:
        text = open(path).read().strip()
        out[path.rsplit("/", 1)[-1]] = {"text": text, "ids": ids(tok, text)}
    for i, c in enumerate(CASES):
        out[f"case.{i}"] = {"text": c, "ids": ids(tok, c)}
    json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
    print({k: len(v["ids"]) for k, v in out.items()})


main()
