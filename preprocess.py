print("*** IMPORT ***")
import os, sys, shutil
import numpy as np
import sklearn.preprocessing
import re
from pathlib import Path
import json
print("***************")

PRE_SALLY_DIR = os.path.join(Path(__file__).parent, "pre_sally")
POST_SALLY_DIR = os.path.join(Path(__file__).parent, "post_sally")
WORKSPACE = os.path.join(Path(__file__).parent, "workspace")

if os.path.exists(PRE_SALLY_DIR):
    shutil.rmtree(PRE_SALLY_DIR)
if os.path.exists(POST_SALLY_DIR):
    shutil.rmtree(POST_SALLY_DIR)
if os.path.exists(WORKSPACE):
    shutil.rmtree(WORKSPACE)
    
        
def find_files(regex_pattern: str, folder: str):
    pattern = re.compile(regex_pattern)
    matching_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if pattern.match(file):
                matching_files.append(os.path.join(root, file))
    return matching_files
    
def main(path):
    vcc_file = "/mnt/e/NewCrawler/output/extracted/FFmpeg/vcc-features-extracted-all-FFmpeg.jsonl"
    patch_files = find_files(r"^simcom-extracted-all-FFmpeg-start-\d+-end-\d+[.]jsonl$","/mnt/e/NewCrawler/output/extracted/FFmpeg")
    
    print("START")
    out_dir = WORKSPACE
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(PRE_SALLY_DIR):
        os.makedirs(PRE_SALLY_DIR)
    if not os.path.exists(POST_SALLY_DIR):
        os.makedirs(POST_SALLY_DIR)
    if not os.path.exists(WORKSPACE):
        os.makedirs(WORKSPACE)

    def load_jsonl(file):
        data = []
        with open(file, "r") as f:
            for line in f:
                data.append(json.loads(line))
                
        return data

    print("*** MAP DATA ***")
    data = load_jsonl(vcc_file)    
    mapping = {key: [] for key in data[0].keys() if key not in ["author", "files", "label"]}
    for datum in data:
        for key in datum.keys():
            if key in ["author", "files", "label"]:
                continue
            mapping[key].append(datum[key])
    del data
    print("*****************")


    print("*** BUILD SCALERS ***")
    normalisers = dict()
    for feat in mapping.keys():
        if feat == "commit_id":
            continue
        scaler = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=10, output_distribution='uniform'
        )
        scaler.fit(
            np.array(mapping[feat]).reshape(-1, 1)
        )
        normalisers[feat] = lambda x, scaler=scaler: int(
            scaler.transform(np.array(x).reshape(-1, 1)) * 100
        )
    print("*******************")


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


    def commit_to_txt(commit_id: int, patch: str, message: str):
        index = mapping["commit_id"].index(commit_id)
        res = list()

        for feat in mapping.keys():
            if feat == "commit_id":
                continue
            val = mapping[feat][index]
            if val:
                normalised = normalisers[feat](val)
                my_bin = get_first_digit(normalised)
                res.append(f"{feat}::{my_bin}")
        
        # Extract text from patch
        res.extend(patch.split('\n'))
        res.extend(message.split('\n'))
        return ' '.join(res)

    def commit_to_file(commit_id: int, patch: str, message: str, dir_location=PRE_SALLY_DIR):
        string = commit_to_txt(commit_id, patch, message).encode('utf-8', errors='ignore').decode("utf-8")
        with open(os.path.join(dir_location, str(commit_id)), 'w') as f:
            f.write(string + "\n")
        
    def yield_jsonl(file):
        with open(file, "r") as f:
            for line in f:
                yield json.loads(line)
    for patch_file in patch_files:
        for commit in yield_jsonl(patch_file):
            commit_id = commit["commit_id"]
            patch = commit["code_change"]
            message = commit["messages"]
            
            commit_to_file(commit_id, patch, message)
        
    print("*** LINES TO VECTOR ***")
    regex = r"[0-9a-zA-Z]{40}"
    files = find_files(regex, PRE_SALLY_DIR)
    cmd = "sally -i lines -o libsvm --vect_embed bin -d' ' -g tokens " + WORKSPACE + "/* " + WORKSPACE + "/one_line.libsvm"
    
    for file in files:
        file = file.rsplit("/")[-1]
        shutil.copy(os.path.join(PRE_SALLY_DIR, file), WORKSPACE)
        os.system(cmd)
        with open(WORKSPACE + '/one_line.libsvm', 'r') as f:
            post_sally_lines = f.readlines()
        assert len(post_sally_lines) == 1
        
        # label = labels[file]
        new_line = (re.sub('#[^\n]+', "#" + file, post_sally_lines[0])).replace('\n', '')
        # new_line = re.sub('^\S+\s', str(label) + " ", new_line)
        
        
        with open(os.path.join(POST_SALLY_DIR, f"{file}.libsvm"), "w") as f:
            f.write(new_line + "\n")
        
        os.remove(os.path.join(PRE_SALLY_DIR, file))
        os.remove(os.path.join(WORKSPACE, file))
        os.remove(os.path.join(WORKSPACE, "one_line.libsvm"))
    print("***********************")
    
if __name__ == "__main__":
    path = "/mnt/e/NewCrawler/vcc"
    main(path)