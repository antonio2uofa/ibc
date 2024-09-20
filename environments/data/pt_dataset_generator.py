import torch
from tqdm import tqdm
import pandas as pd
import argparse
import os


def get_dataset(df, target_index=9):
    """Get the dataset."""

    window_size = 50
    Observations = []

    actions = []
    targets = []

    target_motion = []

    count = 0

    prev_frame = None
    break_flag = False

    frame = None

    for _, group in tqdm(df.groupby('frame_no')):
        
        if len(group) != 6:
            continue
        try: 
            frame = group[group['id'] == target_index][['pitch_predicted', 'yaw_predicted']].iloc[0].tolist()
        except:
            print("skipping")
            continue
        
        if prev_frame is None:
            prev_frame = frame[:]
            continue

        # add the action
        action = [c - p for p, c in zip(prev_frame, frame)]
        actions.append(action[:])
        
        target_motion.append(frame[:])

        # add the action
        observation = []
        for i, row in group.sort_values(by="id")[['pitch_predicted', 'yaw_predicted']].iterrows():
            observation.extend(row.tolist())
            
        Observations.append(observation[:])
        
            
        # add the target at the end of the window
        if count == window_size - 1:
            count = 0
            while len(targets) != len(actions):
                targets.append(frame[:])
        
        
        count += 1
        prev_frame = frame[:]
        
        if break_flag:
            break    
            
    while len(targets) != len(actions):
        targets.append(frame[:])

    return Observations, actions, targets, target_motion


def concat_datasets(folder_path, data_files, facilitator_idx, save_path):
    
    all_Observations, all_actions, all_targets, all_target_motion = [], [], [], []
    for i, file in enumerate(data_files):
        file_path = folder_path + '/' + file

        print("Processing file: ", file)
        df = pd.read_csv(file_path)
        Observations, actions, targets, target_motion = get_dataset(df, facilitator_idx[i])
        all_Observations.extend(Observations)
        all_actions.extend(actions)
        all_targets.extend(targets)
        all_target_motion.extend(target_motion)

    all_Observations = torch.tensor(all_Observations)
    all_actions = torch.tensor(all_actions)
    all_targets = torch.tensor(all_targets)
    all_target_motion = torch.tensor(all_target_motion)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(all_Observations, os.path.join(save_path, 'observations.pt'))
    torch.save(all_actions, os.path.join(save_path, 'actions.pt'))
    torch.save(all_targets, os.path.join(save_path, 'targets.pt'))
    torch.save(all_target_motion, os.path.join(save_path, 'target_motion.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder_path', type=str, default='/app/raw/raw', help='Path to the folder containing the csv files.')
    args = parser.parse_args() 

    folder_path = args.folder_path   

    file_names = ["20220620_L2a-D4_l2cs.csv", "20220619_L3a-D5_l2cs.csv", 
                  "20220627_L4a-D8_l2cs.csv", "20220629_L4b-D8_l2cs.csv"]
    facilitator_idx = [9, 9, 10, 10]
    save_path = "dataset/all_sessions/"

    concat_datasets(folder_path, file_names, facilitator_idx, save_path)