import os,multiprocessing,argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--select_dir',help="select directory to make subdirectory")
args = parser.parse_args()

def move_to_subdir(filename):
    label = filename.split('.')[0].split('_')
    if len(label)==2:
        label = label[1]
    else:
        return
    target_path = os.path.join(args.select_dir,label)
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    command = "mv "+os.path.join(args.select_dir,filename)+" "+ \
os.path.join(target_path,filename)
    os.system(command)

def main():
    selected_dir = args.select_dir
    file_list = []
    for item in os.scandir(selected_dir):
        file_list.append(item.name)
    pool = multiprocessing.Pool()
    with tqdm(total=len(file_list)) as t:
        for _ in pool.imap_unordered(move_to_subdir,file_list):
            t.update(1)

if __name__ == "__main__":
    main()
