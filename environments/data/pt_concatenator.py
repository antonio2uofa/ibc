import torch
import pandas as pd

def generate_csv(data_dir, output_path):
   # Load the .pt file
    actions = torch.load(data_dir + '/' + '/actions.pt')
    observations = torch.load(data_dir + '/' + '/observations.pt')
    # target_motion = torch.load(data_dir + '/' + '/target_motion.pt')
    targets = torch.load(data_dir + '/' + '/targets.pt')

    # Convert the tensor to a NumPy array
    actions_np = actions.numpy()
    observations_np = observations.numpy()
    targets_np = targets.numpy()

    # Convert the NumPy array to a DataFrame
    actions_df = pd.DataFrame(actions_np)
    observations_df = pd.DataFrame(observations_np)
    targets_df = pd.DataFrame(targets_np)

    # Concatenate DataFrames
    df = pd.concat([targets_df, actions_df, observations_df], axis=1)
    print(df)

    df.columns = ['target_x','target_y','velocity_x','velocity_y','obs_1_x','obs_1_y','obs_2_x','obs_2_y',
                    'obs_3_x','obs_3_y','obs_4_x','obs_4_y','obs_5_x','obs_5_y','obs_fac_x','obs_fac_y']

    # Save the DataFrame as a CSV file
    df.to_csv(output_path, index=False) 

if __name__ == "__main__":
    generate_csv(output_path='/app/test_set_pt/20220630_L4c-D8_l2cs/output.csv', data_dir='/app/test_set_pt/20220630_L4c-D8_l2cs')