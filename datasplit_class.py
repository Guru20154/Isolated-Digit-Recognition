import os
from pathlib import Path
from shutil import copyfile

def create_folder_if_not_exists(folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(folder_path+"/"+"test", exist_ok=True)
        os.makedirs(folder_path+"/"+"train", exist_ok=True)
    except FileExistsError:
        pass


def separate(source,i):
    for filename in os.listdir(source):
        first_split = filename.rsplit("_", 1)[0]
        second_split = first_split.rsplit("_", 1)[0]
        split_dir = "train" if i == 0 else "test"
        src = source + "/" + filename
        des = "classwise_data" + "/" + second_split + "/" + split_dir + "/" + filename
        create_folder_if_not_exists("classwise_data" + "/" + second_split )
        copyfile(src, des)
        
def main():
    train_drec = 'training-spectrograms'
    test_drec = 'testing-spectrograms'
    separate(train_drec,0)
    separate(test_drec,1)
    
if __name__ == '__main__':
    main()