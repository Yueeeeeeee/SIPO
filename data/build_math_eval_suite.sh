#!/bin/bash
# Build the standard math eval suite: AIME24, AIME25, AMC23, MATH-500, Minerva Math,
# OlympiadBench -- one parquet per benchmark, so verl reports val-core/<data_source>/*
# separately for each.
#
# Usage:  bash data/build_math_eval_suite.sh          # -> datasets/math_eval/<name>/test.parquet
#         OUT=/path/to/dir bash data/build_math_eval_suite.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
OUT="${OUT:-datasets/math_eval}"

# hf_name -> local dir. The local dir name is irrelevant to routing: data_source comes from
# `dataset` in the records, which _format_math sets to hf_name.split("/")[1].
# hf_name : local_dir : repeat
#
# `repeat` duplicates each row that many times, so validation with val_kwargs.n=1 gives
# avg@repeat for that benchmark. This is how Beyond-the-80/20-Rule does it
# (math__aime_repeated_32x_960.parquet + val_kwargs.n=1), and the reason is that verl
# applies val_kwargs.n uniformly to every val file -- duplication is the only way to
# sample a 30-problem set 32 times without also sampling a 674-problem set 32 times.
#
# Budget at these factors: 960+960+320+500+272+674 = 3686 generations per validation,
# against 24736 if all six ran at val_kwargs.n=16.
SPECS=(
    "math-ai/aime24:aime24:${REPEAT_AIME24:-32}"
    "math-ai/aime25:aime25:${REPEAT_AIME25:-32}"
    "math-ai/amc23:amc23:${REPEAT_AMC23:-8}"
    "math-ai/math500:math500:${REPEAT_MATH500:-1}"
    "math-ai/minervamath:minervamath:${REPEAT_MINERVA:-1}"
    "math-ai/olympiadbench:olympiadbench:${REPEAT_OLYMPIAD:-1}"
)

for spec in "${SPECS[@]}"; do
    IFS=: read -r HF NAME REPEAT <<< "$spec"
    DIR="$OUT/$NAME"
    echo "=============================================================="
    echo "  $HF -> $DIR"
    echo "=============================================================="
    mkdir -p "$DIR"
    python -m data.load_dataset --dataset_name "$HF" --output_path "$DIR/test.json"
    # Repeat *before* preprocess, so preprocess.py writes the parquet and every eval file
    # ends up with the identical LargeString/LargeList schema it produces
    # (write_rowgrouped_large). Repeating the parquet afterwards via pandas silently
    # downgrades those to String/List, and datasets.concatenate_datasets then refuses to
    # align the val files at load time.
    if [ "$REPEAT" -gt 1 ]; then
        python - "$DIR/test.json" "$REPEAT" <<'PY'
import json
import sys

# load_dataset.py writes through write_hf_to_json -> jsonl_to_array, so this file is a
# pretty-printed JSON *array*, not JSON Lines: "[\n  {...},\n  {...}\n]". Repeating it
# line by line concatenates whole arrays and produces invalid JSON. split_tasks.py, by
# contrast, uses Dataset.to_json and does write JSON Lines -- hence the format sniff.
path, k = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as f:
    text = f.read()

if text.lstrip().startswith("["):
    rows, as_array = json.loads(text), True
else:
    rows, as_array = [json.loads(ln) for ln in text.splitlines() if ln.strip()], False

repeated = rows * k
with open(path, "w", encoding="utf-8") as f:
    if as_array:
        f.write("[\n")
        f.write(",\n".join("  " + json.dumps(r, ensure_ascii=False) for r in repeated))
        f.write("\n]")
    else:
        for r in repeated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"  repeated {len(rows)} rows x{k} -> {len(repeated)} (avg@{k} at val_kwargs.n=1)")
PY
    fi
    # preprocess.py reads train.json and test.json from the same directory. These are
    # eval-only sets, so train.parquet is written and never used.
    cp "$DIR/test.json" "$DIR/train.json"
    python -m data.preprocess --data_source "$DIR"
done

echo
echo "Built:"
TOTAL=0
for spec in "${SPECS[@]}"; do
    IFS=: read -r _ NAME REPEAT <<< "$spec"
    N=$(python -c "import pandas;print(len(pandas.read_parquet('$OUT/$NAME/test.parquet')))")
    TOTAL=$((TOTAL + N))
    printf "  %-16s %5d rows (avg@%-2s)  %s\n" "$NAME" "$N" "$REPEAT" "$OUT/$NAME/test.parquet"
done
echo "  ---"
printf "  %-16s %5d generations per validation at val_kwargs.n=1\n" "TOTAL" "$TOTAL"

# All val parquets are concatenated by datasets.concatenate_datasets at load time, which
# refuses to align differing schemas. Catch that here rather than at the first training run.
python - "$OUT" <<'PY'
import sys, pathlib
import pyarrow.parquet as pq

out = pathlib.Path(sys.argv[1])
schemas = {}
for name in ("aime24", "aime25", "amc23", "math500", "minervamath", "olympiadbench"):
    f = out / name / "test.parquet"
    if f.exists():
        schemas[name] = pq.read_schema(f)
ref_name, ref = next(iter(schemas.items()))
bad = [n for n, sc in schemas.items() if not sc.equals(ref)]
if bad:
    print(f"\nSCHEMA MISMATCH against {ref_name}: {bad}")
    for n in bad:
        for field in schemas[n]:
            r = ref.field(field.name) if field.name in ref.names else None
            if r is None or not field.equals(r):
                print(f"  {n}.{field.name}: {field.type}  !=  {r.type if r else '<missing>'}")
    raise SystemExit("val files cannot be concatenated; rebuild with OVERWRITE=1")
print(f"\nschema check: all {len(schemas)} eval parquets agree")
PY

echo
echo "Pass to the trainer with EVAL_SUITE=1, or by hand:"
printf "  data.val_files=["
for i in "${!SPECS[@]}"; do
    IFS=: read -r _ NAME _ <<< "${SPECS[$i]}"
    [ "$i" -gt 0 ] && printf ","
    printf "%s/%s/test.parquet" "$OUT" "$NAME"
done
printf "]\n"
