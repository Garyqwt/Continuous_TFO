import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from imblearn.over_sampling import RandomOverSampler

def normalize_ratio(X_train, X_test):
    X = np.concatenate((np.array(X_train), np.array(X_test)), axis=0)
    col_max = np.max(X[:,:10], axis=0)
    col_min = np.min(X[:,:10], axis=0)
    for i in range(len(X_train)):
        X_train[i, :10] = np.divide((X_train[i, :10] - col_min), (col_max - col_min))
    for i in range(len(X_test)):
        X_test[i, :10] = np.divide((X_test[i, :10] - col_min), (col_max - col_min))
    return X_train, X_test

def standardize_ratio(X_train, X_test):
    X = np.concatenate((np.array(X_train), np.array(X_test)), axis=0)
    col_mean = np.mean(X[:,:10], axis=0)
    col_std = np.std(X[:,:10], axis=0)
    for i in range(len(X_train)):
        X_train[i, :10] = np.divide((X_train[i, :10] - col_mean), (col_std))
    for i in range(len(X_test)):
        X_test[i, :10] = np.divide((X_test[i, :10] - col_mean), (col_std))
    return X_train, X_test

def prepare_shp_rd(ratio_740, ratio_850, RoR_shp_rd, SaO2_labels_shp_rd, label_encoder=None, val_ratio=0.2, val_st_per=0.8, win_len=1.5, verbose = 0):
    '''
    Use ratio for both WLs and SaO2 from each shp_rd to generate training and validation set
    '''

    # def classify_saturation(value):
    #     if 30 <= value < 50:
    #         return 'moderate'
    #     elif 50 <= value <= 70:
    #         return 'good'
    #     else:
    #         return 'severe'

    def classify_saturation(value):
        if value <= 30:
            return 'hypoxemic'
        else:
            return 'non-hypoxemic'
    
    X, y = [], []
    X_train, X_val, y_train, y_val = [], [], [], []
    idx_time_pair = {}
    SaO2_time = SaO2_labels_shp_rd[0]
    start_time, end_time = SaO2_time[0], SaO2_time[-1]
    time_len = end_time - start_time
    if verbose == 1:
        print(f'SaO2 starts at {start_time/60}min, ends at {end_time/60}min.\nValidation starts at {val_st_per}')
    idx = 0
    for time in SaO2_time:
        if time >= len(ratio_740):
            break
        if not np.isnan(RoR_shp_rd[time, :]).any():
            # idx_time_pair[idx] = time
            sample = np.concatenate((ratio_740[time, :], ratio_850[time, :], RoR_shp_rd[time, :], np.array(time).reshape((1))), axis=0)
            X.append(sample)
            y.append(classify_saturation(SaO2_labels_shp_rd[1][idx]))
            # y.append(SaO2_labels_shp_rd[1][idx]/100)
            idx += 1
    
    ratio_columns = np.array(X)[:,:10]
    RoR_columns = np.array(X)[:,10:15]
    time_column = np.array(X)[:,15].reshape((-1,1))
    # normalize RoR for each shp_rd and detector
    scaler_last_5 = StandardScaler() #MinMaxScaler()
    RoR_columns_norm = scaler_last_5.fit_transform(RoR_columns)
    X = np.concatenate((ratio_columns, RoR_columns_norm, time_column), axis=1)

    val_stt_idx = int(val_st_per * len(X))
    val_end_idx = val_stt_idx + int(val_ratio * len(X))
    train_time, val_time = 0, 0
    for idx, sample in enumerate(X):
        time = sample[-1]
        # time = idx_time_pair[idx]
        if idx < val_end_idx and idx >= val_stt_idx and time > train_time+int(win_len*60//2):
            X_val.append(sample)
            y_val.append(label_encoder.transform([y[idx]]))
            # y_val.append(y[idx])
            val_time = time
            # print(f'idx #{idx} time {time}s: validation')
        elif (idx >= val_end_idx or idx < val_stt_idx) and time > val_time+int(win_len*60//2):
            X_train.append(sample)
            y_train.append(label_encoder.transform([y[idx]]))
            # y_train.append(y[idx])
            train_time = time
            # print(f'idx #{idx} time {time}s: training')
    return X_train, X_val, y_train, y_val

def prepare_dataset(ratio_740_dataset, ratio_850_dataset, RoR_dataset, SaO2_dataset, shp_rd_all, label_encoder=None, val_ratio=0.25, val_st_per=0, verbose=0):
    X_train, X_test, y_train, y_test = [], [], [], []
    X_train_dict, y_train_dict = {}, {}
    X_test_dict, y_test_dict = {}, {}
    test_size, train_size = {}, {}
    num_sample_train, num_sample_val = 0, 0
    val_st_per = int(val_st_per*10)
    for shp_rd in shp_rd_all:
        if verbose == 1:
            print(f'========================={shp_rd}=============================')
        X_train_shp_rd, X_test_shp_rd, y_train_shp_rd, y_test_shp_rd = prepare_shp_rd(ratio_740_dataset[shp_rd], 
                    ratio_850_dataset[shp_rd], RoR_dataset[shp_rd], SaO2_dataset[shp_rd], label_encoder=label_encoder,
                    val_ratio=val_ratio, val_st_per=val_st_per/10, win_len=1.5, verbose=verbose)
        
        if verbose == 1:
            test_ratio = len(X_test_shp_rd)/(len(X_test_shp_rd) + len(X_train_shp_rd))
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>num train: {len(X_train_shp_rd)}; num test: {len(X_test_shp_rd)}; Test ratio is {test_ratio:.2f}')

        X_train += X_train_shp_rd
        X_test += X_test_shp_rd
        y_train += y_train_shp_rd
        y_test += y_test_shp_rd
        val_st_per = (val_st_per + 2)%10

        train_size[shp_rd] = (num_sample_train, num_sample_train+len(X_train_shp_rd))
        test_size[shp_rd] = (num_sample_val, num_sample_val+len(X_test_shp_rd))
        num_sample_train += len(X_train_shp_rd)
        num_sample_val += len(X_test_shp_rd)

    # X_train, X_test = normalize_ratio(np.array(X_train), np.array(X_test))
    X_train, X_test = standardize_ratio(np.array(X_train), np.array(X_test))
    # X_train, y_train = oversample_dataset(X_train, y_train, train_size)
    for shp_rd in shp_rd_all:
        X_test_dict[shp_rd] = X_test[test_size[shp_rd][0]:test_size[shp_rd][1]]
        y_test_dict[shp_rd] = y_test[test_size[shp_rd][0]:test_size[shp_rd][1]]
        X_train_dict[shp_rd] = X_train[train_size[shp_rd][0]:train_size[shp_rd][1]]
        y_train_dict[shp_rd] = y_train[train_size[shp_rd][0]:train_size[shp_rd][1]]
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), X_train_dict, y_train_dict, X_test_dict, y_test_dict

# def oversample(X, y, target_num):
#     current_num = len(y)
#     if current_num >= target_num:
#         return X, y  
#     num_duplicates = np.ceil(target_num / current_num).astype(int)
#     X_resampled = np.tile(X, (num_duplicates, 1))[:target_num,:]
#     y_resampled = np.tile(y, num_duplicates)[:target_num]
#     return X_resampled, y_resampled
    

# def oversample_dataset(X_train, y_train, train_size):
#     X_train_over, y_train_over = [], []
#     target_num = max(b - a for a, b in train_size.values())
#     for shp_rd, size in train_size.items():
#         X = X_train[train_size[shp_rd][0]:train_size[shp_rd][1]]
#         y = y_train[train_size[shp_rd][0]:train_size[shp_rd][1]]
#         X_resampled, y_resampled = oversample(X, y, target_num)
#         X_train_over += list(X_resampled)
#         y_train_over += list(y_resampled)
#     return X_train_over, y_train_over