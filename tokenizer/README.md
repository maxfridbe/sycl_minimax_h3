# Qwen2 tokenizer files

`vocab.json`, `merges.txt` and `tokenizer_config.json` of the Qwen2 / Qwen2.5 byte-level BPE tokenizer (Apache-2.0,
Alibaba Cloud), as ComfyUI ships them in `comfy/text_encoders/qwen25_tokenizer/`. The MiniMax H3 text encoder
(Qwen3-VL 32B) uses them unchanged, plus seven extra special tokens that `h3-core::tokenizer` adds itself (`<d>`,
`</d>`, `<|cutoff|>`, `<|lyrics_start|>`, `<|lyrics_end|>`, `<|caption_start|>`, `<|caption_end|>`).
