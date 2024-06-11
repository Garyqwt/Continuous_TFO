import numpy as np

from TFO_dataset import get_sheep_data
import signal_api

def load_data(shp_list, data_ver):
    """
    This function loads selected sheep in shp_list and return dataset as a dictionary.

    Args:
        shp_list: file path to txt file containing selected sheep.
        data_ver: data version to be loaded ("FHR", "MHR", "iq_demod_optical", etc.)

    Returns:
        dataset: a dictionary using each round name as the key.
                 {'sh_rd_name': PPG_data (np.array)}
    """
    dataset = {}
    with open(shp_list, 'r') as file:
        for line in file:
            line = line.strip()
            if not line.startswith("#"):
                segs = line.split("_")
                sheep, round_number, year_prefix = (segs[0].split('R'))[0], (segs[0].split('R'))[1], segs[1]
                add_info = segs[2] if len(segs)>2 else ''
                df = get_sheep_data(int(sheep[1:]), int(round_number), year_prefix=year_prefix, additional_info=add_info, data_version=data_ver)
                key = sheep+'_R'+round_number+'_'+year_prefix
                if add_info != '':
                    key += ('_'+add_info)
                data_per_round = np.array(df)
                dataset[key] = data_per_round
    return dataset

def load_data_norm(shp_list, norm=True):
    """
    The extension of load_data. Return the normalized derivative PPG data as dictionary.
    {'sh_rd_name: PPG_data (np.array)'}

    Args:
        shp_list: file path to txt file containing selected sheep.

    Returns:
        dataset: a dictionary using each round name as the key.
                 {'sh_rd_name': norm_PPG_data (np.array)}
    """
    PPG_data = load_data(shp_list, data_ver="iq_demod_optical")
    if norm:
        norm_PPG = {}
        for key, item in PPG_data.items(): 
            data_round = []
            for col_index in range(item.shape[1]):
                column = item[:, col_index]
                _, result = signal_api.normalized_derivative(column, step=5)
                data_round.append(result)
            norm_PPG[key] = np.array(data_round).transpose()
            PPG_data = norm_PPG
    return PPG_data

# def main(shp_list, size_sec, step_sec, Fs):
#     # size_sec, step_sec, Fs = 60, 1, 80
#     path = str(Path(__file__).parent.joinpath(shp_list).absolute())
#     norm_PPG, SaO2_data = read_data(path)
#     dataset_dict = prepare_dataset(SaO2_data, norm_PPG, size_sec=size_sec, step_sec=step_sec, Fs=Fs)
#     dataset_export(dataset_dict, size_sec, step_sec, Fs)

# if __name__ == '__main__':
#     default_shp_list = 'shp_list.txt'
#     parser = argparse.ArgumentParser()
#     parser.add_argument("shp_list", help="Sheep list to load data")
#     parser.add_argument("-size", "--size_sec", default=60, type=int, help="sample size in second")
#     parser.add_argument("-step", "--step_sec", default=1, type=int, help="sliding step in second")
#     parser.add_argument("-fs", "--fs", default=80, type=int, help="sampling frequency")
#     args = parser.parse_args()
#     main(args.shp_list, args.size_sec, args.step_sec, args.fs)