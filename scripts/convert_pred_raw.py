#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def collect_files(paths):
    files = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("results/*.jsonl")))
    return files


def rewrite_file(src, suffix):
    out_path = src.with_name(f"{src.stem}{suffix}{src.suffix}")
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite {out_path}")
    with src.open() as src_f, out_path.open("w") as dst_f:
        for line in src_f:
            if not line.strip():
                continue
            obj = json.loads(line)
            raw = obj.pop("pred_raw", None)
            if raw is not None:
                obj["pred"] = raw
            else:
                obj.pop("pred", None)
            dst_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Copy JSONL and promote pred_raw to pred")
    parser.add_argument("entries", nargs="+", help="File or directory to process")
    parser.add_argument(
        "--suffix",
        "-s",
        default="_pred_raw_as_pred",
        help="Suffix to append before the extension",
    )
    args = parser.parse_args()

    files = collect_files(args.entries)
    if not files:
        raise SystemExit("No JSONL files found")

    for file_path in files:
        out_file = rewrite_file(file_path, args.suffix)
        print(f"Created {out_file}")


if __name__ == "__main__":
    main()

