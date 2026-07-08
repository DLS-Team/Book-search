from datasets import get_dataset_config_names, load_dataset

name = "zkeown/gutenberg-corpus"
configs = get_dataset_config_names(name)
print("configs:", configs)

for cfg in configs:
    print("\nCONFIG:", cfg)
    try:
        ds = load_dataset(name, cfg, split="train")
        print("rows:", len(ds))
        print("columns:", ds.column_names)
        print(ds[0])
    except Exception as e:
        print("ERROR:", e)