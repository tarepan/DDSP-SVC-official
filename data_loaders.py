import os
import random
import numpy as np
import librosa
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset

def traverse_dir(
        root_dir,
        extension,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


def get_data_loaders(args, whole_audio=False):
    data_train = AudioDataset(
        args.data.train_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=whole_audio,
        n_spk=args.model.n_spk,
        device=args.train.cache_device)
    loader_train = torch.utils.data.DataLoader(
        data_train ,
        batch_size=args.train.batch_size if not whole_audio else 1,
        shuffle=True,
        num_workers=args.train.num_workers if args.train.cache_device=='cpu' else 0,
        persistent_workers=(args.train.num_workers > 0) if args.train.cache_device=='cpu' else False,
        pin_memory=True if args.train.cache_device=='cpu' else False
    )
    data_valid = AudioDataset(
        args.data.valid_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        n_spk=args.model.n_spk)
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=False,
        num_workers=args.train.num_workers,
        persistent_workers=(args.train.num_workers > 0),
        pin_memory=True
    )
    return loader_train, loader_valid 


class AudioDataset(Dataset):
    def __init__(self,
        path_root,                  # Path of the directory under which preprocessed features exist
        waveform_sec,
        hop_size,
        sample_rate: int,           # Audio sampling rate, with which audio will be loaded
        load_all_data: bool = True, # How many data loaded on device (True: all data, False: only light-weight items)
        whole_audio=False,          # Whether to use whole audio or w/ length clipping
        n_spk: int = 1,             # The number of speakers
        device = 'cpu'              # Device on which cache data will be loaded
    ):
        super().__init__()
        
        self.waveform_sec = waveform_sec
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.path_root = path_root
        # Path of .wav files
        self.paths = traverse_dir(os.path.join(path_root, 'audio'), extension='wav', is_pure=True, is_sort=True, is_ext=False)
        self.whole_audio = whole_audio
        self.data_buffer={}
        if load_all_data:
            print('Load all the data from :', path_root)
        else:
            print('Load the f0, volume data from :', path_root)
        for name in tqdm(self.paths, total=len(self.paths)):
            path_audio = os.path.join(self.path_root, 'audio', name) + '.wav'

            # Audio::(T,) / Unit::maybe(Frame, Feat) (if configured, on device)
            audio  = torch.from_numpy(librosa.load(path_audio, sr=self.sample_rate, mono=True)[0]   ).float().to(device) if load_all_data else None
            units  = torch.from_numpy(np.load(os.path.join(self.path_root, 'units',  name) + '.npy')).float().to(device) if load_all_data else None
            # Duration::(1,) / fo::maybe(Frame, 1) / Volume::(Frame, 1) - always on device
            duration = librosa.get_duration(filename = path_audio, sr = self.sample_rate)
            f0     = torch.from_numpy(np.load(os.path.join(self.path_root, 'f0',     name) + '.npy')).float().unsqueeze(-1).to(device)
            volume = torch.from_numpy(np.load(os.path.join(self.path_root, 'volume', name) + '.npy')).float().unsqueeze(-1).to(device)

            # Speaker index :: (1,) (always on device)
            if n_spk is not None and n_spk > 1:
                spk_id = int(os.path.dirname(name)) if str.isdigit(os.path.dirname(name)) else 0
                if spk_id < 1 or spk_id > n_spk:
                    raise ValueError(' [x] Muiti-speaker traing error : spk_id must be an integer within [1, n_spk]')
            else:
                spk_id = 1
            spk_id = torch.LongTensor(np.array([spk_id])).to(device)

            # Pack
            self.data_buffer[name] = { 'audio': audio, 'units': units, 'duration': duration, 'f0': f0, 'volume': volume, 'spk_id': spk_id, }

    def __getitem__(self, file_idx):
        name = self.paths[file_idx]
        data_buffer = self.data_buffer[name]
        # check duration. if too short, then skip
        if data_buffer['duration'] < (self.waveform_sec + 0.1):
            return self.__getitem__( (file_idx + 1) % len(self.paths))
            
        # get item
        return self.get_data(name, data_buffer)

    def get_data(self, name, data_buffer):
        """
        Returns:
            dict
                audio  :: (T,)
                f0     :: (Frame, 1) maybe - [Hz]
                volume :: (Frame, 1)
                units  :: (Frame, Feat) maybe
                spk_id :: (1,)
                name   :: - self.paths[file_idx]
        """

        # Load
        audio  = data_buffer.get('audio') # nullable
        units  = data_buffer.get('units') # nullable
        f0     = data_buffer.get('f0')
        volume = data_buffer.get('volume')
        spk_id = data_buffer.get('spk_id')
        if units is None:
            units = torch.from_numpy(np.load(os.path.join(self.path_root, 'units', name) + '.npy')).float()

        # Clipping
        ## parameters
        frame_resolution = self.hop_size / self.sample_rate
        duration = data_buffer['duration']
        waveform_sec = duration if self.whole_audio else self.waveform_sec
        idx_from = 0 if self.whole_audio else random.uniform(0, duration - waveform_sec - 0.1)
        start_frame = int(idx_from / frame_resolution)
        units_frame_len = int(waveform_sec / frame_resolution)
        ## audio (with partial load)
        if audio is None:
            path_audio = os.path.join(self.path_root, 'audio', name) + '.wav'
            audio = torch.from_numpy(
                librosa.load(path_audio, sr = self.sample_rate, mono=True, offset = start_frame * frame_resolution, duration = waveform_sec)[0]
            ).float()
            # clip audio into N seconds
            audio = audio[ : audio.shape[-1] // self.hop_size * self.hop_size]
        else:
            audio = audio[start_frame * self.hop_size : (start_frame + units_frame_len) * self.hop_size]
        ## unit/fo/volume
        unit_frames   =  units[start_frame : start_frame + units_frame_len]
        f0_frames     =     f0[start_frame : start_frame + units_frame_len]
        volume_frames = volume[start_frame : start_frame + units_frame_len]

        return dict(audio=audio, f0=f0_frames, volume=volume_frames, units=unit_frames, spk_id=spk_id, name=name)

    def __len__(self):
        return len(self.paths)
