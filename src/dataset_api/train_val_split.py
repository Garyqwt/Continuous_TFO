import dataset_api
import yaml, pickle
from pathlib import Path

def train_val_idx_split(cfg_path, validation_ratio=0.2, shift=0.2):
    # Get the number of samples for each ship_rd
    dataset_info = dataset_api.import_dataset(cfg_path, w_data=False)
    samples_per_round = {}
    for item in dataset_info:
        num_samples = item['num_samples']
        samples_per_round[item['name']] = num_samples

    train_data_idx = {}
    val_data_idx = {}
    split_frac = 0
    for name, size in samples_per_round.items():
        split_idx_strt = int((split_frac % 1)*size)
        split_idx_end = int(((split_frac % 1)+validation_ratio)*size)

        val_idx = list(range(split_idx_strt+9,split_idx_end-9))
        train_idx = [i for i in range(size) if i not in range(split_idx_strt, split_idx_end)]
        train_data_idx[name] = train_idx
        val_data_idx[name] = val_idx
        split_frac += shift
    return train_data_idx, val_data_idx

def train_val_idx_export(train_data_idx, val_data_idx, export_path):
    split_train_path = export_path + 'split_train_idx.txt'
    split_val_path = export_path + 'split_val_idx.txt'
    filename = str(Path(__file__).parent.joinpath(split_train_path).absolute())
    with open(filename, 'w') as file:
        for key, item in train_data_idx.items():
            file.write(f"\\\{key}\n")
            idx = ', '.join(str(num) for num in item.X)
            file.write(idx + '\n')

    filename = str(Path(__file__).parent.joinpath(split_val_path).absolute())
    with open(filename, 'w') as file:
        for key, item in val_data_idx.items():
            file.write(f"\\\{key}\n")
            idx = ', '.join(str(num) for num in item.X)
            file.write(idx + '\n')

    print('Tran-val index splitted and exported.')


