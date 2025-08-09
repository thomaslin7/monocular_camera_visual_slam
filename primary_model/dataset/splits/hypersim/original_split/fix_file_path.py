import os

# Directory containing the files
files = ["train.txt", "val.txt"]

# Prefix to remove
prefix = "/mnt/bn/liheyang/DepthDatasets/"

for file_name in files:
    with open(file_name, "r") as f:
        lines = f.readlines()
    
    # Strip the prefix from both paths in each line
    new_lines = [line.replace(prefix, "") for line in lines]
    
    # Overwrite the file with the modified lines
    with open(file_name, "w") as f:
        f.writelines(new_lines)

print("Prefixes removed from train.txt and val.txt ✅")
