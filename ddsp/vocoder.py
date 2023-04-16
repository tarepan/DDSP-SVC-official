import os
import numpy as np
import yaml
import torch
import pyworld as pw
import parselmouth
import torchcrepe
import resampy
# from fairseq import checkpoint_utils
from encoder.hubert.model import HubertSoft
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torchaudio.transforms import Resample
from .unit2control import Unit2Control
from .core import fo_to_rot, frequency_filter, upsample, remove_above_fmax, MaskedAvgPool1d, MedianPool1d


CREPE_RESAMPLE_KERNEL = {}

class F0_Extractor:
    def __init__(self, f0_extractor: str, sample_rate: int = 44100, hop_size: int = 512, f0_min: int = 65, f0_max: int = 800):
        """
        Args:
            f0_extractor - 'parselmouth' | 'dio' | 'harvest' | 'crepe'
            sample_rate
            hop_size
            f0_min
            f0_max
        """
        self.f0_extractor, self.sample_rate, self.hop_size, self.f0_min, self.f0_max = f0_extractor, sample_rate, hop_size, f0_min, f0_max
        if f0_extractor == 'crepe':
            key_str = str(sample_rate)
            if key_str not in CREPE_RESAMPLE_KERNEL:
                CREPE_RESAMPLE_KERNEL[key_str] = Resample(sample_rate, 16000, lowpass_filter_width = 128)
            self.resample_kernel = CREPE_RESAMPLE_KERNEL[key_str]
    
    def extract(self, audio, uv_interp: bool = False, device = None, silence_front = 0):
        """
        Args:
            audio :: (T,)
            uv_interp - Whether to interpolated fo==0 unvoiced region with voiced edge fo values
            device    - PyTorch device (only for 'crepe')
        Returns:
            f0 - [Hz]
        """
        # extractor start time
        n_frames = int(len(audio) // self.hop_size) + 1

        start_frame = int(silence_front * self.sample_rate / self.hop_size)
        real_silence_front = start_frame * self.hop_size / self.sample_rate
        audio = audio[int(np.round(real_silence_front * self.sample_rate)) : ]

        # 'parselmouth' - parselmouth by `parselmouth`
        if self.f0_extractor == 'parselmouth':
            f0 = parselmouth.Sound(audio, self.sample_rate).to_pitch_ac(
                time_step = self.hop_size / self.sample_rate, 
                voicing_threshold = 0.6,
                pitch_floor = self.f0_min, 
                pitch_ceiling = self.f0_max).selected_array['frequency']
            pad_size = start_frame + (int(len(audio) // self.hop_size) - len(f0) + 1) // 2
            f0 = np.pad(f0,(pad_size, n_frames - len(f0) - pad_size))

        # 'dio' - DIO + stonemask by `pyworld`
        elif self.f0_extractor == 'dio':
            _f0, t = pw.dio(audio.astype('double'), self.sample_rate, f0_floor = self.f0_min, f0_ceil = self.f0_max, channels_in_octave=2,
                frame_period = (1000 * self.hop_size / self.sample_rate))
            f0 = pw.stonemask(audio.astype('double'), _f0, t, self.sample_rate)
            f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))

        # 'harvest' - Harvest by `pyworld`
        elif self.f0_extractor == 'harvest':
            f0, _ = pw.harvest(
                audio.astype('double'), 
                self.sample_rate, 
                f0_floor = self.f0_min, 
                f0_ceil = self.f0_max, 
                frame_period = (1000 * self.hop_size / self.sample_rate))
            f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))

        # 'crepe' - CREPE by 'torchcrepe'
        elif self.f0_extractor == 'crepe':
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            resample_kernel = self.resample_kernel.to(device)
            wav16k_torch = resample_kernel(torch.FloatTensor(audio).unsqueeze(0).to(device))

            f0, pd = torchcrepe.predict(wav16k_torch, 16000, 80, self.f0_min, self.f0_max, pad=True, model='full', batch_size=512, device=device, return_periodicity=True)
            pd = MedianPool1d(pd, 4)
            f0 = torchcrepe.threshold.At(0.05)(f0, pd)
            f0 = MaskedAvgPool1d(f0, 4)
            
            f0 = f0.squeeze(0).cpu().numpy()
            f0 = np.array([f0[int(min(int(np.round(n * self.hop_size / self.sample_rate / 0.005)), len(f0) - 1))] for n in range(n_frames - start_frame)])
            f0 = np.pad(f0, (start_frame, 0))

        else:
            raise ValueError(f" [x] Unknown f0 extractor: {self.f0_extractor}")

        # interpolate unvoiced (fo==0) f0 region
        if uv_interp:
            uv = (f0 == 0)
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            f0[f0 < self.f0_min] = self.f0_min
        return f0


class Volume_Extractor:
    def __init__(self, hop_size = 512):
        """
        Args:
            hop_size - Hop size defining RMS frame width
        """
        self.hop_size = hop_size
        
    def extract(self, audio):
        """Extract non-overlapped RMS series from an audio :: (T,) -> (Frame,)"""
        n_frames = int(len(audio) // self.hop_size) + 1

        # Padding for centering ...?
        audio = np.pad(audio, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode = 'reflect')

        # Sample-wise Square
        audio2 = audio ** 2
        # Frame-wise Mean of Square
        volume = np.array([np.mean(audio2[int(n * self.hop_size) : int((n + 1) * self.hop_size)]) for n in range(n_frames)])
        # Frame-wise Root of MeanSquare
        volume = np.sqrt(volume)
        return volume


class Units_Encoder:
    def __init__(self, encoder: str, encoder_ckpt, encoder_sample_rate = 16000, encoder_hop_size = 320, device = None):
        """
        Args:
            encoder             - Encoder type specifier
            encoder_ckpt
            encoder_sample_rate - Audio sampling rate of Encoder input
            encoder_hop_size    - Hop size of encoder output relative to Encoder input audio
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        if   encoder == 'hubertsoft':
            self.model = Audio2HubertSoft(encoder_ckpt).to(device)
        elif encoder == 'hubertbase':
            self.model = Audio2HubertBase(encoder_ckpt, device=device)
        elif encoder == 'hubertbase768':
            self.model = Audio2HubertBase768(encoder_ckpt, device=device)
        elif encoder == 'contentvec':
            self.model = Audio2ContentVec(encoder_ckpt, device=device)
        elif encoder == 'contentvec768':
            self.model = Audio2ContentVec768(encoder_ckpt, device=device)
        elif encoder == 'xunit':
            self.model = Audio2XUnit(device=device)
        elif encoder == 'yunit':
            self.model = Audio2YUnit(device=device)
        else:
            raise ValueError(f" [x] Unknown units encoder: {encoder}")
        self.model = self.model.eval()

        # { f"{sample_rate}": Resample_instance}
        self.resample_kernel = {}
        self.encoder_sample_rate = encoder_sample_rate
        self.encoder_hop_size = encoder_hop_size
        
    def encode(self, audio, sample_rate: int, hop_size: int):
        """
        Args:
            audio :: (B, T) - Input audio
            sample_rate     - Input sampling rate
            hop_size        - Desired hop size of unit relative to `audio`
              in preprocess, `d.block_size`
              in main, `args.data.block_size * sr_i_librosa / args.data.sampling_rate`
        """

        # Resample for encoder
        if sample_rate == self.encoder_sample_rate:
            # pass
            audio_res = audio
        else:
            key_str = str(sample_rate)
            # Resample instance
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.encoder_sample_rate, lowpass_filter_width = 128).to(self.device)
            # Resampling
            audio_res = self.resample_kernel[key_str](audio)

        # wave-to-unit :: (B, T) -> (B, Frame, Feat)
        units = self.model(audio_res)

        # alignment - Nearest interpolation (align unit to audio (≠resampled))
        ## There is no grantee of well-shaped unit space, so use nearest
        n_frames = audio.size(-1) // hop_size + 1
        raw_unit_frame_period_sec = self.encoder_hop_size / self.encoder_sample_rate
        target_unit_frame_period_sec = hop_size / sample_rate
        ratio = target_unit_frame_period_sec / raw_unit_frame_period_sec
        # [0, 1, 2, ... , N-1] -> 1.5 * [0, 1, 2, ... , N-1] -> (round&clip) -> [0, 2, 3, ...]
        frame_index = torch.clamp(torch.round(ratio * torch.arange(n_frames).to(self.device)).long(), max = units.size(1) - 1)
        #                                 FrameDim   (Frame) -> (1, Frame, Feat)
        units_aligned = torch.gather(units, 1, frame_index.unsqueeze(0).unsqueeze(-1).repeat([1, 1, units.size(-1)]))
        return units_aligned


class Audio2HubertSoft(torch.nn.Module):
    def __init__(self, path, h_sample_rate = 16000, h_hop_size = 320):
        super().__init__()
        print(' [Encoder Model] HuBERT Soft')
        self.hubert = HubertSoft()
        print(' [Loading] ' + path)
        checkpoint = torch.load(path)
        consume_prefix_in_state_dict_if_present(checkpoint, "module.")
        self.hubert.load_state_dict(checkpoint)
        self.hubert.eval()
     
    def forward(self, audio):
        """ :: (B, T) -> (B, 1, T) -> (B, Frame, Feat=256) """
        with torch.inference_mode():
            return self.hubert.units(audio.unsqueeze(1))


class Audio2ContentVec():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] Content Vec')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert.eval()

    def __call__(self, audio):  # B, T
        # wav_tensor = torch.from_numpy(audio).to(self.device)
        wav_tensor = audio
        feats = wav_tensor.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.to(wav_tensor.device),
            "padding_mask": padding_mask.to(wav_tensor.device),
            "output_layer": 9,  # layer 9
        }
        with torch.no_grad():
            logits = self.hubert.extract_features(**inputs)
            feats = self.hubert.final_proj(logits[0])
        units = feats  # .transpose(2, 1)
        return units


class Audio2ContentVec768():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] Content Vec')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        # wav_tensor = torch.from_numpy(audio).to(self.device)
        wav_tensor = audio
        feats = wav_tensor.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.to(wav_tensor.device),
            "padding_mask": padding_mask.to(wav_tensor.device),
            "output_layer": 9,  # layer 9
        }
        with torch.no_grad():
            logits = self.hubert.extract_features(**inputs)
            feats = logits[0]
        units = feats  # .transpose(2, 1)
        return units


class Audio2HubertBase():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] HuBERT Base')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert = self.hubert.float()
        self.hubert.eval()

    def __call__(self, audio):  # B, T
        with torch.no_grad():
            padding_mask = torch.BoolTensor(audio.shape).fill_(False)
            inputs = {
                "source": audio.to(self.device),
                "padding_mask": padding_mask.to(self.device),
                "output_layer": 9,  # layer 9
            }
            logits = self.hubert.extract_features(**inputs)
            units = self.hubert.final_proj(logits[0])
            return units


class Audio2HubertBase768():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] HuBERT Base')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert = self.hubert.float()
        self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        with torch.no_grad():
            padding_mask = torch.BoolTensor(audio.shape).fill_(False)
            inputs = {
                "source": audio.to(self.device),
                "padding_mask": padding_mask.to(self.device),
                "output_layer": 9,  # layer 9
            }
            logits = self.hubert.extract_features(**inputs)
            units = logits[0]
            return units


class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__
    
def load_model(model_path, device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    model = None
    if args.model.type == 'Sins':
        model = Sins(sampling_rate=args.data.sampling_rate, block_size=args.data.block_size,
                        n_harmonics=args.model.n_harmonics, n_mag_allpass=args.model.n_mag_allpass, n_mag_noise=args.model.n_mag_noise,
                        n_unit=args.data.encoder_out_channels, n_spk=args.model.n_spk, c=args.model.c,)
    elif args.model.type == 'CombSub':
        model = CombSub(sampling_rate=args.data.sampling_rate, block_size=args.data.block_size,
                        n_mag_allpass=args.model.n_mag_allpass, n_mag_harmonic=args.model.n_mag_harmonic, n_mag_noise=args.model.n_mag_noise,
                        n_unit=args.data.encoder_out_channels, n_spk=args.model.n_spk, c=args.model.c,)
    elif args.model.type == 'CombSubFast':
        model = CombSubFast(sampling_rate=args.data.sampling_rate, block_size=args.data.block_size,
                        n_unit=args.data.encoder_out_channels, n_spk=args.model.n_spk, c=args.model.c,)
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")

    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, args


class Sins(torch.nn.Module):
    def __init__(self, sampling_rate, block_size, n_harmonics, n_mag_allpass, n_mag_noise, n_unit=256, n_spk=1, c: bool=False):
        super().__init__()
        print(' [DDSP Model] Sinusoids Additive Synthesiser')

        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.unit2ctrl = Unit2Control(n_unit, n_spk, { 'amplitudes': n_harmonics, 'group_delay': n_mag_allpass, 'noise_magnitude': n_mag_noise, }, c)

    def forward(self, units_frames, f0_frames, volume_frames, spk_id, spk_mix_dict=None, initial_phase=None, infer=True, max_upsample_dim=32):
        '''
            units_frames  :: (B, Frame, Feat) - Unit series
            f0_frames     :: (B, Frame, 1)
            volume_frames :: (B, Frame)       - Frame-wise volume contour (non-overlapped RMS of the waveform)
            spk_id        :: (B,)             - Speaker index

            initial_phase :: (B,)             - Initial phase [rad] (not used now)
        '''
        # Sample/Frame-wise frequency contour [Hz] & phase contour [rad] :: (B, T) | (B, Frame)
        f0 = upsample(f0_frames, self.block_size).squeeze(-1)
        phase = 2 * np.pi * fo_to_rot(f0, self.sampling_rate, initial_phase, infer)
        phase_frames = phase[:, ::self.block_size]

        # parameter prediction
        ctrls = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk_id, spk_mix_dict=spk_mix_dict)
        amplitudes_frames =    torch.exp(ctrls['amplitudes']) / 128
        group_delay = np.pi * torch.tanh(ctrls['group_delay'])
        noise_param =          torch.exp(ctrls['noise_magnitude']) / 128
        
        # sinusoids exciter signal 
        amplitudes_frames = remove_above_fmax(amplitudes_frames, f0_frames, self.sampling_rate / 2, level_start = 1)
        n_harmonic = amplitudes_frames.shape[-1]
        level_harmonic = torch.arange(1, n_harmonic + 1).to(phase)
        sinusoids = 0.
        for n in range(( n_harmonic - 1) // max_upsample_dim + 1):
            start = n * max_upsample_dim
            end = (n + 1) * max_upsample_dim
            phases = phase.unsqueeze(-1) * level_harmonic[start:end]
            amplitudes = upsample(amplitudes_frames[:,:,start:end], self.block_size)
            # A * sin(phase)
            sinusoids += (amplitudes * torch.sin(phases)).sum(-1)

        # harmonic part filter (apply group-delay)
        harmonic = frequency_filter(sinusoids, torch.exp(1.j * torch.cumsum(group_delay, axis = -1)), hann_window = False)
                        
        # SubNoise 
        noise = torch.rand_like(harmonic) * 2 - 1
        noise = frequency_filter(noise, torch.complex(noise_param, torch.zeros_like(noise_param)), hann_window = True)
                        
        signal = harmonic + noise

        return signal, phase.unsqueeze(-1), (harmonic, noise)


class CombSubFast(torch.nn.Module):
    def __init__(self, sampling_rate, block_size, n_unit=256, n_spk=1, c: bool=False):
        super().__init__()
        print(' [DDSP Model] Combtooth Subtractive Synthesiser')

        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size",    torch.tensor(block_size))
        # Half-overlap window
        self.register_buffer("window",        torch.sqrt(torch.hann_window(2 * block_size)))
        self.unit2ctrl = Unit2Control(n_unit, n_spk, { 'harmonic_magnitude': block_size + 1, 'harmonic_phase': block_size + 1, 'noise_magnitude': block_size + 1, }, c)

    def forward(self, units_frames, f0_frames, volume_frames, spk_id, spk_mix_dict=None, initial_phase=None, infer=True, **kwargs):
        '''aNN (conformer) + sDSP (SubHarmo + SubNoise + OLA)
            units_frames  :: (B, Frame, Feat)    - Unit series
            f0_frames     :: (B, Frame, 1) maybe - Fundamental tone's frequency series [Hz]
            volume_frames :: (B, Frame)          - Frame-wise volume contour (non-overlapped RMS of the waveform)
            spk_id        :: (B,)                - Speaker index

            initial_phase :: (B,)             - Initial phase [rad] (not used now)
        '''
        pad_l, pad_r = self.block_size, self.block_size

        # Sample/Frame-wise frequency contour [Hz] & rotation contour [・] & phase contour [rad] :: (B, T) | (B, Frame)
        f0 = upsample(f0_frames, self.block_size).squeeze(-1)
        rot = fo_to_rot(f0, self.sampling_rate, initial_phase, infer)
        phase_frames = 2 * np.pi * rot[:, ::self.block_size]

        # parameter prediction
        ctrls = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk_id, spk_mix_dict=spk_mix_dict)
        harmo_mag, harmo_phase, noise_mag = ctrls['harmonic_magnitude'], ctrls['harmonic_phase'], ctrls['noise_magnitude']

        # Excitations
        ## combtooth - rotation contour [Hz] / fo contour [Hz]
        combtooth = torch.sinc(self.sampling_rate * rot / (f0 + 1e-3))
        combtooth[f0<=0.] = 0.
        noise     = torch.rand_like(combtooth) * 2 - 1
        ## Framing - Padding & unfolding & windowing
        combtooth = F.pad(combtooth, (pad_l, pad_r))
        noise     = F.pad(noise,     (pad_l, pad_r))
        combtooth_frames = combtooth.unfold(1, 2 * self.block_size, self.block_size)
        noise_frames     =     noise.unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        noise_frames     = noise_frames     * self.window

        # filters - (maybe) transfer function H(ω)
        ## harmo - H_h(ω) = exp(a + j*π*p)
        _src_filter = torch.exp(harmo_mag + 1.j * np.pi * harmo_phase)
        src_filter = torch.cat((_src_filter, _src_filter[:,-1:,:]), 1)
        ## noise - H_n(ω) = 1/128 * exp(x + 0j)
        _noise_filter = torch.exp(noise_mag) / 128
        noise_filter = torch.cat((_noise_filter, _noise_filter[:,-1:,:]), 1)

        # apply the filters in frequency domain
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        noise_fft     = torch.fft.rfft(noise_frames,     2 * self.block_size)
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter
        signal_frames_out = torch.fft.irfft(signal_fft,  2 * self.block_size)

        # OverLap-Add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal_frames_out = signal_frames_out * self.window
        signal = fold(signal_frames_out.transpose(1, 2))

        # Remove padding
        signal = signal[:, 0, 0, pad_l:-pad_r]

        return signal, phase_frames.unsqueeze(-1), (signal, signal)


class CombSub(torch.nn.Module):
    def __init__(self, sampling_rate, block_size, n_mag_allpass, n_mag_harmonic, n_mag_noise, n_unit=256, n_spk=1, c: bool=False):
        super().__init__()
        print(' [DDSP Model] Combtooth Subtractive Synthesiser (Old Version)')

        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.unit2ctrl = Unit2Control(n_unit, n_spk, { 'group_delay': n_mag_allpass, 'harmonic_magnitude': n_mag_harmonic, 'noise_magnitude': n_mag_noise, }, c)

    def forward(self, units_frames, f0_frames, volume_frames, spk_id, spk_mix_dict=None, initial_phase=None, infer=True, **kwargs):
        '''
            units_frames  :: (B, Frame, Feat) - Unit series
            f0_frames     :: (B, Frame, 1)    - fo series
            volume_frames :: (B, Frame)       - Frame-wise volume contour (non-overlapped RMS of the waveform)
            spk_id        :: (B,)             - Speaker index
            spk_mix_dict
            initial_phase :: (B,)             - Initial phase [rad] (not used now)
            infer
        '''
        # Sample/Frame-wise frequency contour [Hz] & rotation contour [・] & phase contour [rad] :: (B, T) | (B, Frame)
        f0 = upsample(f0_frames, self.block_size).squeeze(-1)
        rot = fo_to_rot(f0, self.sampling_rate, initial_phase, infer)
        phase_frames = 2 * np.pi * rot[:, ::self.block_size]
        
        # parameter prediction
        ctrls = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk_id, spk_mix_dict=spk_mix_dict)
        group_delay = np.pi * torch.tanh(ctrls['group_delay'])
        src_param   =          torch.exp(ctrls['harmonic_magnitude'])
        noise_param =          torch.exp(ctrls['noise_magnitude']) / 128

        """
                                (`f0_frames`?) ----------------------------
                                                                           |
                     ----------> `src_param` -------------------------------
                    |                                                       |
              [NN] -|            `combtooth`  ---> [SubComb] -> `harmonic` ---> [SubHarmo] -> `harmonic` - [+] -> `signal`
                    |                          |                                                          |
                    |---------> `group_delay` -                                                           |
                    |                                                                                     |
                     ----> ---> `noise_param` ---> [SubNoise] -> `noise` ---------------------------------
                                                      |
                                                rand -
        """
        # SubHarmo (dynamic-windowed LTV-FIR, with group-delay prediction)
        combtooth = torch.sinc(self.sampling_rate * rot / (f0 + 1e-3))
        harmonic = frequency_filter(combtooth, torch.exp(1.j * torch.cumsum(group_delay, axis = -1)), hann_window = False)
        harmonic = frequency_filter(harmonic,  torch.complex(src_param, torch.zeros_like(src_param)), hann_window = True,
                                                        half_width_frames = 1.5 * self.sampling_rate / (f0_frames + 1e-3))

        # SubNoise (constant-windowed LTV-FIR, without group-delay)
        noise = torch.rand_like(harmonic) * 2 - 1
        noise = frequency_filter(noise, torch.complex(noise_param, torch.zeros_like(noise_param)), hann_window = True)
                        
        signal = harmonic + noise

        return signal, phase_frames.unsqueeze(-1), (harmonic, noise)
