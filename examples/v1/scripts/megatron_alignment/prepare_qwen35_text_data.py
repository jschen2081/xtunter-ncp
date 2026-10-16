"""Normalize original JSONL text for xTuner without creating token datasets."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Iterator


def _records(path: Path) -> Iterator[dict]:
    if path.name.endswith(".jsonl.zst"):
        import zstandard

        with path.open("rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as stream:
                with io.TextIOWrapper(stream, encoding="utf-8") as reader:
                    for line in reader:
                        if line.strip():
                            yield json.loads(line)
    else:
        with path.open(encoding="utf-8") as reader:
            for line in reader:
                if line.strip():
                    yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    files = ([source] if source.is_file() else sorted(
        p for p in source.rglob("*") if p.name.endswith((".jsonl", ".jsonl.zst"))))
    if not files or any(not p.name.endswith((".jsonl", ".jsonl.zst")) for p in files):
        raise ValueError("DATA_PATH must be a JSONL/JSONL.zst file or a directory containing such text files")
    # Cache text/schema conversion only. Tokenization, packing and sampling are performed by xTuner.
    key = hashlib.sha256(str(source).encode()).hexdigest()[:16]
    target = args.cache_root / key
    target.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in files:
        stat = path.stat()
        identity = f"{path}:{stat.st_size}:{stat.st_mtime_ns}:content-v1"
        name = hashlib.sha256(identity.encode()).hexdigest()[:24] + ".jsonl"
        output = target / name
        if not output.exists():
            temporary = target / (name + f".{os.getpid()}.tmp")
            with temporary.open("w", encoding="utf-8") as writer:
                for row in _records(path):
                    if "content" in row:
                        item = {"content": row["content"]}
                    elif "text" in row:
                        item = {"content": row["text"]}
                    elif "messages" in row:
                        item = {"messages": row["messages"]}
                    else:
                        raise ValueError(f"No content, text or pretrain messages in {path}")
                    writer.write(json.dumps(item, ensure_ascii=False) + "\n")
            temporary.replace(output)
        manifest.append(dict(source=str(path), normalized=str(output)))
    # Use only this manifest's files; stale cache files must not become extra datasets.
    (target / "sources.json").write_text(json.dumps(manifest, indent=2))
    print(target / "sources.json")


if __name__ == "__main__":
    main()
