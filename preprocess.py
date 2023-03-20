import os
import numpy as np
import librosa
import torch
import argparse
import shutil
from logger import utils
from tqdm import tqdm
from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from logger.utils import traverse_dir


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)


def preprocess(path, f0_extractor, volume_extractor, units_encoder, sample_rate, hop_size, device = 'cuda'):

    # Directories
    path_srcdir     = os.path.join(path, 'audio')
    path_unitsdir   = os.path.join(path, 'units')
    path_f0dir      = os.path.join(path, 'f0')
    path_volumedir  = os.path.join(path, 'volume')
    path_skipdir    = os.path.join(path, 'skip')

    # run  
    def process(file):
        ext = file.split('.')[-1]
        binfile = file[:-(len(ext)+1)]+'.npy'
        path_srcfile    = os.path.join(path_srcdir,    file)    # Path of source audio                           (.wav)
        path_unitsfile  = os.path.join(path_unitsdir,  binfile) # Path of preprocessed unit                      (.npy)
        path_f0file     = os.path.join(path_f0dir,     binfile) # Path of preprocessed fo                        (.npy)
        path_volumefile = os.path.join(path_volumedir, binfile) # Path of preprocessed volume                    (.npy)
        path_skipfile   = os.path.join(path_skipdir,   file)    # Path to which audio is moved when all unvoiced (.wav)

        # Audio :: NDArray
        audio = librosa.load(path_srcfile, sr=sample_rate, mono=True)[0]

        # Volume :: NDArray
        volume = volume_extractor.extract(audio)

        # Unit :: NDArray
        units = units_encoder.encode(torch.from_numpy(audio).float().unsqueeze(0).to(device), sample_rate, hop_size).squeeze().to('cpu').numpy()

        # fo :: NDArray
        f0 = f0_extractor.extract(audio, uv_interp = False)
        unvoiced = (f0 == 0)
        if len(f0[~unvoiced]) > 0:
            # contain voiced, so interpolate the unvoiced f0
            f0[unvoiced] = np.interp(np.where(unvoiced)[0], np.where(~unvoiced)[0], f0[~unvoiced])

        # Save
            os.makedirs(os.path.dirname(path_unitsfile),  exist_ok=True)
            os.makedirs(os.path.dirname(path_f0file),     exist_ok=True)
            os.makedirs(os.path.dirname(path_volumefile), exist_ok=True)
            np.save(path_unitsfile,  units)
            np.save(path_f0file,     f0)
            np.save(path_volumefile, volume)
        else:
            # all unvoiced, skip (move to skip directory)
            print('\n[Error] F0 extraction failed: ' + path_srcfile)
            os.makedirs(os.path.dirname(path_skipfile),   exist_ok=True)
            shutil.move(path_srcfile, os.path.dirname(path_skipfile))
            print('This file has been moved to ' + path_skipfile)


    print('Preprocess the audio clips in :', path_srcdir)
    filelist = traverse_dir(path_srcdir, extension='wav', is_pure=True, is_sort=True, is_ext=True)
    for file in tqdm(filelist, total=len(filelist)):
        process(file)


if __name__ == '__main__':

    # Configs
    args = utils.load_config(parse_args().config)
    d = args.data

    # Init
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    f0_extractor = F0_Extractor(d.f0_extractor, d.sampling_rate, d.block_size, d.f0_min, d.f0_max)
    volume_extractor = Volume_Extractor(d.block_size)
    units_encoder = Units_Encoder(d.encoder, d.encoder_ckpt, d.encoder_sample_rate, d.encoder_hop_size, device = device)

    # Preprocess train/val
    preprocess(d.train_path, f0_extractor, volume_extractor, units_encoder, d.sampling_rate, d.block_size, device = device)
    preprocess(d.valid_path, f0_extractor, volume_extractor, units_encoder, d.sampling_rate, d.block_size, device = device)
