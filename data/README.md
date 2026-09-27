# Data Generation

Run every command below from the repository root.

## Benchmark datasets

* LiveCodeBench v6: `python -m data.load_dataset --dataset_name livecodebench/code_generation_lite-v6 --output_path datasets/lcb_v6.json`

## Generalization Experiments

* tooluse: the json files are already provided in `datasets/tooluse/train.json` and `datasets/tooluse/test.json`
* sciknoweval Biology: `python -m data.load_dataset --dataset_name Biology --output_path datasets/sciknoweval/biology/biology.json`
* sciknoweval Chemistry: `python -m data.load_dataset --dataset_name Chemistry --output_path datasets/sciknoweval/chemistry/chemistry.json`
* sciknoweval Material: `python -m data.load_dataset --dataset_name Material --output_path datasets/sciknoweval/material/material.json`
* sciknoweval Physics: `python -m data.load_dataset --dataset_name Physics --output_path datasets/sciknoweval/physics/physics.json`


## LiveCodeBench

Create a train/test split of the tests by running:
```bash
python -m data.split_tests \
    --json_path datasets/lcb_v6.json \
    --output_dir datasets/lcb_v6
```

## SciKnowEval
Create a train/test split of the data by running:
```bash
python -m data.split_tasks \
    --json_path datasets/sciknoweval/biology/biology.json \
    --output_dir datasets/sciknoweval/biology \
    --test_ratio 0.1 --seed 42

python -m data.split_tasks \
    --json_path datasets/sciknoweval/chemistry/chemistry.json \
    --output_dir datasets/sciknoweval/chemistry \
    --test_ratio 0.1 --seed 42

python -m data.split_tasks \
    --json_path datasets/sciknoweval/material/material.json \
    --output_dir datasets/sciknoweval/material \
    --test_ratio 0.1 --seed 42

python -m data.split_tasks \
    --json_path datasets/sciknoweval/physics/physics.json \
    --output_dir datasets/sciknoweval/physics \
    --test_ratio 0.1 --seed 42
```

## Preprocessing
Our implementation uses the `parquet` format for the data. To preprocess the data, run the following command:
```bash
python -m data.preprocess \
    --data_source DATASET_PATH
```
`DATASET_PATH` should contain the `train.json` and `test.json` files.

