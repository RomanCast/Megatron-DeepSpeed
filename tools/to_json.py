import argparse
import jsonlines
import time
import sys
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--size", type=int, default=None)
    args = parser.parse_args()

    all_files = sorted(args.inputs, key=lambda x: (len(x), x))

    t0 = time.time()
    text = ""
    print("Reading files...", flush=True)
    for filename in tqdm(all_files):
        with open(filename, "r") as f:
            text += f.read()

    all_lines = []
    print("Creating jsonlines...", flush=True)
    for line in tqdm(text.split("\n")):
        all_lines.append({"text": line})
        if args.size is not None and sys.getsizeof(all_lines) / (1024 ** 3) > args.size:
            break

    with jsonlines.open(args.out, mode="w") as writer:
        writer.write_all(all_lines)

    t1 = time.time()

    print(f"Done in {t1-t0}s.", flush=True)

    

if __name__ == "__main__":
    main()
