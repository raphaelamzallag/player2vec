import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import json

def load_all_events_data(dataset_path='../raw_data', sub_dir='sub', verbose=True):
    data = []
    if verbose:
        print('\nLoading all events data')
    dir_ = os.path.join(dataset_path, sub_dir, '')
    files_ = os.listdir(dir_)
    for match_ in tqdm(files_, total=len(files_)):
        with open(f'{dir_}{match_}') as data_file:
            data_ = json.load(data_file)
            data.append(pd.json_normalize(data_, sep="_").assign(match_id=match_))
    if verbose:
        print(' - COMPLETED\n')
    all_events_data = pd.concat(data)
    return all_events_data

data = load_all_events_data()
df = data.copy()

for y in ["Starting XI","Half Start","Pressure","Camera On","Camera off","Tactical Shift","Offside","Substitution","Injury Stoppage",
 "Referee Ball-Drop","Player On","Player Off"]:
    df = df[df.type_name != y]
