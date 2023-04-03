import os
import torch
import librosa
import argparse
import numpy as np
import soundfile as sf
import pyworld as pw
import parselmouth
from ast import literal_eval
from slicer import Slicer
from ddsp.vocoder import load_model, F0_Extractor, Volume_Extractor, Units_Encoder
from ddsp.core import upsample
from enhancer import Enhancer
from tqdm import tqdm


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m",    "--model_path",      type=str, required=True,                   help="path to the model file")
    parser.add_argument("-i",    "--input",           type=str, required=True,                   help="path to the input audio file")
    parser.add_argument("-o",    "--output",          type=str, required=True,                   help="path to the output audio file")
    parser.add_argument("-id",   "--spk_id",          type=str, required=False, default=1,       help="speaker id (for multi-speaker model) | default: 1")
    parser.add_argument("-mix",  "--spk_mix_dict",    type=str, required=False, default="None",  help="mix-speaker dictionary (for multi-speaker model) | default: None")
    parser.add_argument("-k",    "--key",             type=str, required=False, default=0,       help="key changed (number of semitones) | default: 0")
    parser.add_argument("-e",    "--enhance",         type=str, required=False, default='true',  help="true or false | default: true")
    parser.add_argument("-pe",   "--pitch_extractor", type=str, required=False, default='crepe', help="pitch extrator type: parselmouth, dio, harvest, crepe (default)")
    parser.add_argument("-fmin", "--f0_min",          type=str, required=False, default=50,      help="min f0 (Hz) | default: 50")
    parser.add_argument("-fmax", "--f0_max",          type=str, required=False, default=1100,    help="max f0 (Hz) | default: 1100")
    parser.add_argument("-th",   "--threhold",        type=str, required=False, default=-60,     help="response threhold (dB) | default: -60")
    parser.add_argument("-eak", "--enhancer_adaptive_key", type=str, required=False, default=0,  help="adapt the enhancer to a higher vocal range (number of semitones) | default: 0")
    parser.add_argument("-sr",  "--sampling_rate",    type=int, required=False, default=44100,   help="Audio sampling rate")
    return parser.parse_args(args=args, namespace=namespace)
    
def split(audio, sample_rate, hop_size, db_thresh = -40, min_len = 5000):
    slicer = Slicer(sr=sample_rate, threshold=db_thresh, min_length=min_len)       
    chunks = dict(slicer.slice(audio))
    result = []
    for _, v in chunks.items():
        tag = v["split_time"].split(",")
        if tag[0] != tag[1]:
            start_frame = int(int(tag[0]) // hop_size)
            end_frame = int(int(tag[1]) // hop_size)
            if end_frame > start_frame:
                result.append((
                        start_frame, 
                        audio[int(start_frame * hop_size) : int(end_frame * hop_size)]))
    return result


def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(dst=result[:idx], src=a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx: a.shape[0]] = (1 - k) * a[idx:] + k * b[: fade_len]
    np.copyto(dst=result[a.shape[0]:], src=b[fade_len:])
    return result


if __name__ == '__main__':
    #device = 'cpu' 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # parse commands
    cmd = parse_args()
    
    # load ddsp model
    model, args = load_model(cmd.model_path, device=device)
    
    # load input :: (T,)
    audio, sr_i = librosa.load(cmd.input, sr=cmd.sampling_rate, mono=True)
    hop_size = args.data.block_size * sr_i / args.data.sampling_rate
    
    # Analysis
    ## fo
    print('Pitch extractor type: ' + cmd.pitch_extractor)
    pitch_extractor = F0_Extractor(cmd.pitch_extractor, sr_i, hop_size, float(cmd.f0_min), float(cmd.f0_max))
    # NOTE: in flask_api.py, silence_front is used
    f0 = torch.from_numpy(pitch_extractor.extract(audio, uv_interp = True, device = device)).float().to(device).unsqueeze(-1).unsqueeze(0)
    ## Volume :: (T,) -> (B=1, Frame)
    volume_np = Volume_Extractor(hop_size).extract(audio)
    volume = torch.from_numpy(volume_np).float().to(device).unsqueeze(0)
    ## mask
    mask = (volume_np > 10 ** (float(cmd.threhold) / 20)).astype('float')
    mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
    mask = np.array([np.max(mask[n : n + 9]) for n in range(len(mask) - 8)])
    mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
    mask = upsample(mask, args.data.block_size).squeeze(-1)

    # Modification
    ## fo - key change
    f0 = f0 * 2 ** (float(cmd.key) / 12)
    ## Speaker - speaker id or mix-speaker dictionary
    spk_mix_dict = literal_eval(cmd.spk_mix_dict)
    if spk_mix_dict is not None:
        print('Mix-speaker mode')
    else:
        print('Speaker ID: '+ str(int(cmd.spk_id)))        
    spk_id = torch.LongTensor(np.array([[int(cmd.spk_id)]])).to(device)

    # load units encoder
    units_encoder = Units_Encoder(args.data.encoder, args.data.encoder_ckpt, args.data.encoder_sample_rate, args.data.encoder_hop_size, device = device)

    # load enhancer
    if cmd.enhance == 'true':
        print('Enhancer type: ' + args.enhancer.type)
        enhancer = Enhancer(args.enhancer.type, args.enhancer.ckpt, device=device)
    else:
        print('Enhancer type: none (using raw output of ddsp)')

    # forward and save the output
    # NOTE: in flask_api.py, unit series are extracted at first as batch, then synth waveform at once.
    result = np.zeros(0)
    current_length = 0
    segments = split(audio, sr_i, hop_size)
    print('Cut the input audio into ' + str(len(segments)) + ' slices')
    with torch.no_grad():
        for segment in tqdm(segments):
            start_frame = segment[0]
            seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(device)
            # Analysis
            ## Unit :: -> (B, Frame, Feat)
            seg_units = units_encoder.encode(seg_input, sr_i, hop_size)
            ## fo/Volume
            seg_f0     =     f0[:, start_frame : start_frame + seg_units.size(1), :]
            seg_volume = volume[:, start_frame : start_frame + seg_units.size(1)]
            
            # Synthesis
            ## DDSP
            seg_output = model(seg_units, seg_f0, seg_volume, spk_id = spk_id, spk_mix_dict = spk_mix_dict)[0]
            seg_output *= mask[:, start_frame * args.data.block_size : (start_frame + seg_units.size(1)) * args.data.block_size]
            ## Enhancer
            if cmd.enhance == 'true':
                seg_output, sr_o = enhancer.enhance(seg_output, args.data.sampling_rate, seg_f0, args.data.block_size, adaptive_key = float(cmd.enhancer_adaptive_key))
            else:
                sr_o = args.data.sampling_rate
            seg_output = seg_output.squeeze().cpu().numpy()
            
            silent_length = round(start_frame * args.data.block_size * sr_o / args.data.sampling_rate) - current_length
            if silent_length >= 0:
                result = np.append(result, np.zeros(silent_length))
                result = np.append(result, seg_output)
            else:
                result = cross_fade(result, seg_output, current_length + silent_length)
            current_length = current_length + silent_length + len(seg_output)
        sf.write(cmd.output, result, sr_o)
    