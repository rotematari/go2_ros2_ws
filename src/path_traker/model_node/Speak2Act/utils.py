import os
import time
import datetime
import re
import torch
import numpy as np
import ast
import pandas as pd
import pickle
from tqdm import tqdm
#!/usr/bin/env python3

def gpu_to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def errors_cm(outputs, targets):
    sh_targets = targets[:, :3]
    sh_outputs = outputs[:, :3]
    location_error_sh = np.linalg.norm(sh_targets - sh_outputs, axis=1).mean()
    return location_error_sh


def save_net(path, state, epoch):
    tt = str(time.asctime())
    img_name_save = epoch + '_' + 'net' + " " + str(re.sub('[:!@#$]', '_', tt))
    img_name_save = img_name_save.replace(' ', '_') + '.pt'
    _dir = os.path.abspath('../')
    path = os.path.join(_dir, path)
    t = datetime.datetime.now()
    datat = t.strftime('%m/%d/%Y').replace('/', '_')
    dir = os.path.join(path, datat)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir, exist_ok=True)
            print("Directory '%s' created successfully" % ('net' + '_' + datat))
        except OSError as error:
            print("Directory '%s' can not be created" % ('net' + '_' + datat))

    net_path = os.path.join(dir, img_name_save)
    print()
    print(net_path)
    torch.save(state, net_path)
    return net_path


def weights_name(full_path):
    splited_path = full_path.split('/')
    return splited_path[-1]


def save_summary(config, weights_name, summary):
    file_name = "summary_" + weights_name + ".txt"
    file_path = os.path.join(config.path4summary, config.model_name, file_name)
    with open(file_path, 'w') as file:
        file.write(summary)


def save_epoch_summary(config, weight_name, summary):
    weight_name = weight_name.split('/')[-1].replace('.pt', '')
    file_name = "summary_epoch_" + weight_name  + ".txt"
    file_path = os.path.join(config.path4summary, config.model_name, file_name)
    with open(file_path, 'w') as file:
        file.write(summary)


def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()


def load_files(path):
    with open(path, 'rb') as file:
        loaded_data = pickle.load(file)
    df = pd.DataFrame(loaded_data)
    return df


# def load_files_and_concat_old(path):
#     data_list = []
#     # files_list = []
#     for root, directories, files in os.walk(path):
#         for att in files:
#             if att.endswith('.pkl'):
#                 file_path = os.path.join(root, att)
#                 with open(file_path, 'rb') as file:
#                     loaded_data = pickle.load(file)
#                 df = pd.DataFrame(loaded_data)
#                 # files_list.append(att)
#                 data_list.append(df)
#     data = pd.concat(data_list, axis=0, ignore_index=True)
#     return data


def remove_duplicates(arrays):
    seen = set()
    unique_arrays = []
    for arr in arrays:
        arr_tuple = tuple(arr)
        if arr_tuple not in seen:
            seen.add(arr_tuple)
            unique_arrays.append(arr)

    unique_arrays = np.array(unique_arrays)
    return unique_arrays


def chunk_data(data):
    # Identify the rows where the 'id' column is 0
    reset_indices = data.index[data['id'] == 0].tolist()
    chunks = []
    # Add the end index of the DataFrame to the list
    for i, idx in enumerate(reset_indices):
        if idx == reset_indices[-1]:
            chunk = data.iloc[idx:, :].reset_index(drop=True)
        else:
            chunk = data.iloc[idx:reset_indices[i+1], :].reset_index(drop=True)
        chunks.append(chunk)
    return chunks


def new_load_files_and_chunk_data(path):
    df_list = []
    with tqdm() as pbar:
        for root, directories, files in os.walk(path):
            for att in files:
                if att.endswith('.pkl'):
                    if len(df_list) < 73: # 73, 14, 40
                        pbar.set_description(f"Processing {att}")
                        pbar.update(1)
                        file_path = os.path.join(root, att)
                        with open(file_path, 'rb') as file:
                            loaded_data = pickle.load(file)
                        df = pd.DataFrame(loaded_data)
                        df_list.append(df)
                    else:
                        break
    data = pd.concat(df_list, axis=0, ignore_index=True)
    command_list = data['command'].tolist()
    pos_list = data['position'].tolist()
    or_list = data['orientation'].tolist()
    time_list = data['timestamp'].tolist()
    return command_list, pos_list, or_list, time_list

def preprocess_string(s):
    s = s.strip('[]')
    s = ','.join(s.split())
    s = f'[{s}]'
    return s

def convert_strings_to_arrays(strings):
    arrays = [np.array(ast.literal_eval(preprocess_string(s))) for s in strings]
    return arrays

def get_input_output(chunks):
    input_list = []
    pos_list = []
    or_list = []
    time_list = []

    for i, data_chunk in enumerate(chunks):
        if not data_chunk.empty and len(data_chunk) > 0:

            input_list.append(data_chunk['command'].iloc[0])
            pos = data_chunk['position'].tolist()
            if all(isinstance(elem, str) for elem in pos):
                pos = convert_strings_to_arrays(pos)
            pos_list.append(pos)

            ori = data_chunk['orientation'].tolist()
            if all(isinstance(elem, str) for elem in ori):
                ori = convert_strings_to_arrays(ori)
            or_list.append(ori)

            time = data_chunk['timestamp'].tolist()
            time_list.append(time)

    return input_list, pos_list, or_list, time_list


def save_aug_data(df, data_path, original_pkl_file_name, addon=None, with_csv=False):
    day = datetime.datetime.now()
    day = day.strftime('%m/%d/%Y').replace('/', '_')
    _dir = os.path.join(data_path, day)
    if not os.path.exists(_dir):
        os.mkdir(_dir)

    file_name = os.path.splitext(os.path.basename(original_pkl_file_name))[0]

    if addon:
        file_name += addon

    pkl_name_save = file_name + "_aug.pkl"
    csv_file_name = file_name + "_aug.csv"

    pkl_file_path = os.path.join(_dir, pkl_name_save)
    csv_file_path = os.path.join(_dir, csv_file_name)

    data_dict = df.to_dict(orient='list')

    # Save the dictionary to a pickle file
    with open(pkl_file_path, 'wb') as file:
        pickle.dump(data_dict, file)

    if with_csv:
        df.to_csv(csv_file_path, index=False)

    print()
    print("Data path: ", pkl_file_path)
    print()


def save_clean_data(data_dict, original_pkl_file_name, addon=None, with_csv=False):
    outer_dir = os.path.dirname(os.path.dirname(original_pkl_file_name))
    new_root = outer_dir.replace('/data/', '/clean_data/')
    date = original_pkl_file_name.split('/')[-2]

    _outer_dir = os.path.join(new_root,date)
    if not os.path.exists(_outer_dir):
        os.mkdir(_outer_dir)

    file_name = os.path.splitext(os.path.basename(original_pkl_file_name))[0]

    if addon:
        file_name += addon

    pkl_name_save = file_name + "_aug.pkl"
    pkl_file_path = os.path.join(_outer_dir, pkl_name_save)

    with open(pkl_file_path, 'wb') as file:
        pickle.dump(data_dict, file)

    if with_csv:
        csv_file_name = file_name + "_aug.csv"
        csv_file_path = os.path.join(_outer_dir, csv_file_name)
        df = pd.DataFrame(data_dict)
        df.to_csv(csv_file_path, index=False)

    print()
    print("Data path: ", pkl_file_path)
    print()


def quaternion_to_yaw(qz, qw):
    sin_y_cos_p = 2 * (qw * qz)
    cosy_cos_p = 1 - 2 * (qz * qz)
    yaw = np.arctan2(sin_y_cos_p, cosy_cos_p)
    return yaw