import numpy as np
import tqdm
import os
import shutil
import wave

WAV_MIN_LENGTH = 2 # The minimum duration of wav files
SAMPLE_RATE = 1    # The percentage of files to be extracted
SAMPLE_MIN = 2     # The lower limit of the number of files to be extracted
SAMPLE_MAX = 10    # The upper limit of the number of files to be extracted


def check_duration(wav_file):
    f = wave.open(wav_file, "rb")
    # 获取帧数和帧率
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    f.close()

    return duration > WAV_MIN_LENGTH


# 定义一个函数，用于从给定的目录中随机抽取一定比例的wav文件，并剪切到另一个目录中，保留数据结构
def split_data(src_dir, dst_dir, ratio):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # ソースディレクトリ以下のすべてのサブディレクトリとファイル名を取得する（サブディレクトリの内容は除く）。
    # 获取源目录下所有的子目录和文件名（不包括子目录下的内容）
    subdirs, files = [], []
    for item in os.listdir(src_dir):
        item_path = os.path.join(src_dir, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
        elif os.path.isfile(item_path) and item.endswith(".wav"):
            files.append(item)

    if len(files) == 0:
        print(f"Error: No wav files found in {src_dir}")
        return

    # Randomly select splitted files
    num_files = int(len(files) * ratio)
    # SAMPLE_MIN < num_files < SAMPLE_MAX
    num_files = max(SAMPLE_MIN, min(SAMPLE_MAX, num_files))
    np.random.shuffle(files)
    selected_files = files[:num_files]

    # Split by Move
    pbar = tqdm.tqdm(total=num_files)
    for file in selected_files:
        src_file = os.path.join(src_dir, file)
        dst_file = os.path.join(dst_dir, file)
        # length check
        if check_duration(src_file):
            shutil.move(src_file, dst_file)
            pbar.update(1)
        else:
            print(f"Skipped {src_file} because its duration is less than 2 seconds.")
    pbar.close()

    # recursive split
    for subdir in subdirs:
        src_subdir = os.path.join(src_dir, subdir)
        dst_subdir = os.path.join(dst_dir, subdir)
        split_data(src_subdir, dst_subdir, ratio)


def main():
    root_dir = os.path.abspath('.')
    dst_dir = root_dir + "/data/val/audio"
    # 抽取比例，默认为1
    ratio = float(SAMPLE_RATE) / 100

    src_dir = root_dir + "/data/train/audio"

    # 调用split_data函数，对源目录中的wav文件进行抽取，并剪切到目标目录中，保留数据结构
    split_data(src_dir, dst_dir, ratio)

# 如果本模块是主模块，则执行主函数
if __name__ == "__main__":
    main()