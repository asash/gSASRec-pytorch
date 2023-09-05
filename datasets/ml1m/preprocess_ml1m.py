from collections import defaultdict
import requests
from pathlib import Path
import numpy as np
import json

#we use pre-processed version of the dataset from SASRec repository
#in this version of the datasets items with less than 5 interactions are removed
#the same version of the dataset is used in gSASRec and BERT4Rec papers
#https://github.com/asash/ml1m-sas-mapping
#each line is a single user-item interaction in format "<userid> <itemid>"
#for each user the interactions are orderered chronologically
#original dataset had some number of ties, and unfortunately we don't know how they were handled

URL="https://raw.githubusercontent.com/asash/ml1m-sas-mapping/main/ml-1m_sas.txt"

DATASET_DIR = Path(__file__).parent
TRAIN_DIR = DATASET_DIR/"train"
VAL_DIR = DATASET_DIR/"val"
TEST_DIR = DATASET_DIR/"test"
FILE_NAME = DATASET_DIR/"ml-1m.txt"


def download():
    if not Path(FILE_NAME).exists():
        request = requests.get(URL, timeout=10, stream=True)
        # Open the output file and make sure we write in binary mode
        with open(FILE_NAME, 'wb') as fh:
            # Walk through the request response in chunks of 1024 * 1024 bytes, so 1MiB
            for chunk in request.iter_content(1024 * 1024):
                # Write the chunk to the file
                fh.write(chunk)
                # Optionally we can check here if the download is taking too long

#gSASRec paper used last interaction of each user for testing
#for 512 users, it also used the second last interaction for validation

def train_val_test_split():
    TRAIN_DIR.mkdir(exist_ok=True)
    VAL_DIR.mkdir(exist_ok=True)
    TEST_DIR.mkdir(exist_ok=True)

    rng = np.random.RandomState(42)
    val_users = rng.choice(6040, 512, replace=False)
    user_items = defaultdict(list) 
    num_interactions = 0
    items = set()
    with open(FILE_NAME) as f:
        for line in f:
            if len(line.strip().split(" ")) != 2:
                continue
            user, item = line.strip().split(" ")
            user = int(user)
            item = int(item)
            items.add(item)
            num_interactions += 1
            user_items[user].append(item)
    dataset_stats = {
                    "num_users": len(user_items),
                    "num_items": len(items), 
                    "num_interactions": num_interactions
                     }

    print("Dataset stats: ", json.dumps(dataset_stats, indent=4))
    with open(DATASET_DIR/"dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=4)

    train_sequences = []

    val_input_sequences = []
    val_gt_actions = []

    test_input_sequences = []
    test_gt_actions = []


    for user in user_items:
        if user in val_users:
            train_input_sequence = user_items[user][:-3]
            train_sequences.append(train_input_sequence)

            val_input_sequence = user_items[user][:-2] 
            val_gt_action = user_items[user][-2]
            val_input_sequences.append(val_input_sequence)
            val_gt_actions.append(val_gt_action)

            test_input_sequence = user_items[user][:-1]
            test_input_sequences.append(test_input_sequence)

            test_gt_action = user_items[user][-1]
            test_gt_actions.append(test_gt_action)
        else:
            train_input_sequence = user_items[user][:-2]
            train_sequences.append(train_input_sequence)

            test_input_sequence = user_items[user][:-1]
            test_input_sequences.append(test_input_sequence)
            test_gt_action = user_items[user][-1]
            test_gt_actions.append(test_gt_action) 

    with open(TRAIN_DIR/"input.txt", "w") as f:
        for sequence in train_sequences:
            f.write(" ".join([str(item) for item in sequence]) + "\n")

    with open(VAL_DIR/"input.txt", "w") as f:
        for sequence in val_input_sequences:
            f.write(" ".join([str(item) for item in sequence]) + "\n")
    
    with open(VAL_DIR/"output.txt", "w") as f:
        for action in val_gt_actions:
            f.write(str(action) + "\n")

    with open(TEST_DIR/"input.txt", "w") as f:
        for sequence in test_input_sequences:
            f.write(" ".join([str(item) for item in sequence]) + "\n")
    with open(TEST_DIR/"output.txt", "w") as f:
        for action in test_gt_actions:
            f.write(str(action) + "\n")

if __name__ == "__main__":
    download()
    train_val_test_split()