import torch
from tqdm import tqdm
import pandas as pd
import argparse
import os

def read_csv(path):
    return pd.read_csv(path)

def get_dataset(df):
    """Get the dataset."""

    window_size = 50
    target_index = 10

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

    Observations = torch.tensor(Observations)
    actions = torch.tensor(actions)
    targets = torch.tensor(targets)
    target_motion = torch.tensor(target_motion)

    return Observations, actions, targets, target_motion

def save_data(file_path):
    save_path = os.path.join("dataset", os.path.basename(file_path).split('.')[0])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df = read_csv(file_path)
    Observations, actions, targets, target_motion = get_dataset(df)
    torch.save(Observations, os.path.join(save_path, 'observations.pt'))
    torch.save(actions, os.path.join(save_path, 'actions.pt'))
    torch.save(targets, os.path.join(save_path, 'targets.pt'))
    torch.save(target_motion, os.path.join(save_path, 'target_motion.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, default='data/data.csv')
    args = parser.parse_args()    

    save_data(args.file_path)