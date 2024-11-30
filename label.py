print("*** IMPORT ***")
import os, sys, shutil
import numpy as np
import sklearn.preprocessing
import re
from pathlib import Path
import json
from tqdm import tqdm
print("***************")

PRE_SALLY_DIR = os.path.join(Path(__file__).parent, "pre_sally")
POST_SALLY_DIR = os.path.join(Path(__file__).parent, "post_sally")
WORKSPACE = os.path.join(Path(__file__).parent, "workspace")


def get_first_digit(x):
    """
    Value -> bin value
    10 possible bins
    x should be < 100
    """
    x = int(x)
    if x < 0:
        return 0
    x = str(x)
    if len(x) == 1:  # less than 10 ?
        return 0
    else:
        return int(x[0])
    
def yield_jsonl(file):
    with open(file, "r") as f:
        for line in f:
            yield json.loads(line)
    
def load_jsonl(file):
    data = []
    with open(file, "r") as f:
        for line in f:
            data.append(json.loads(line))
            
    return data

def main(label_file, target_file):
    print("*** MAP DATA ***")
    labels = {}
    # for vcc_file in vcc_files:
    data = load_jsonl(label_file)    
    for datum in data:
        labels[datum["commit_id"]] = datum["label"]
    del data
    print("*****************")
    
    print("*** LINES TO VECTOR ***")
    with open(target_file, "r+") as f:
        lines = f.readlines()
        new_lines = []
        for line in tqdm(lines):
            id = line.rsplit("#")[-1]
            label = labels[file]
            new_lines.append(re.sub('^\S+\s', str(label) + " ", line))
        
        f.seek(0)
        f.writelines(new_lines)
    print("***********************")
    
if __name__ == "__main__":
    setup = "SETUP1"
    project = "linux"
    path = f"../data/linux/{setup}"
    
    label_files = [
        f"{path}/{setup}-{project}-vcc-features-val.jsonl",
        f"{path}/{setup}-{project}-vcc-features-test.jsonl",
        f"{path}/unsampling/{setup}-{project}-vcc-features-train.jsonl"
    ]
    
    target_files = [
        f"post_sally/{setup}-{project}-vcc-features-val.libsvm",
        f"post_sally/{setup}-{project}-vcc-features-test.libsvm",
        f"post_sally/{setup}-{project}-vcc-features-train.libsvm"
    ] 
    
    for l, t in zip(label_files, target_files):
        main(l, t)