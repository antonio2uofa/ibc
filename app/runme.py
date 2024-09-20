import tensorflow as tf
import pandas as pd

# Define the feature description according to the structure
feature_description = {
    'reward': tf.io.FixedLenFeature([], tf.float32),
    'step_type': tf.io.FixedLenFeature([], tf.string),
    'next_step_type': tf.io.FixedLenFeature([], tf.string),
    'discount': tf.io.FixedLenFeature([], tf.float32),
    'action': tf.io.FixedLenFeature([2], tf.float32),  # Assuming action is a 2D array
    'observation/fac_gaze_agent': tf.io.FixedLenFeature([4], tf.float32),  # Assuming 2D velocity
    'observation/all_gaze_agent': tf.io.FixedLenFeature([24], tf.float32),  # Assuming 2D position
    'observation/pos_first_goal': tf.io.FixedLenFeature([4], tf.float32),  # Assuming 2D position
    'observation/vel_gaze_agent': tf.io.FixedLenFeature([4], tf.float32),  # Assuming 2D position
}

# Function to parse each example
def parse_example(example_proto):
    parsed_data = tf.io.parse_single_example(example_proto, feature_description)

    # Decode the step_type and next_step_type from the first byte of the string
    parsed_data['step_type'] = tf.io.decode_raw(parsed_data['step_type'], tf.uint8)[0]
    parsed_data['next_step_type'] = tf.io.decode_raw(parsed_data['next_step_type'], tf.uint8)[0]

    return parsed_data

# Load the TFRecord file
tfrecord_file = "/app/ibc/data/gaze/evaluation/2d_L3ad5_data.tfrecord"
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

# Parse the dataset and save to CSV
rows = []
for raw_record in raw_dataset:
    parsed_record = parse_example(raw_record)
    rows.append(parsed_record)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(rows)
df.to_csv("/app/eval_output.csv", index=False)


