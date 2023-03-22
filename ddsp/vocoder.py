import os
import numpy as np
import yaml
import torch
import pyworld as pw
import parselmouth
import torchcrepe
import resampy
from encoder.hubert.model import HubertSoft
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torchaudio.transforms import Resample
from .unit2control import Unit2Control
from .core import frequency_filter, upsample, remove_above_fmax


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
    
    def extract(self, audio, uv_interp: bool = False, device = None, silence_front = 0): # audio: 1d numpy array
        """
        Args:

            uv_interp - Whether to interpolated fo==0 unvoiced region with voiced edge fo values
            device    - PyTorch device (only for 'crepe')
        """
        # extractor start time
        n_frames = int(len(audio) // self.hop_size) + 1

        start_frame = int(silence_front * self.sample_rate / self.hop_size)
        real_silence_front = start_frame * self.hop_size / self.sample_rate
        audio = audio[int(np.round(real_silence_front * self.sample_rate)) : ]

        # parselmouth by 'parselmouth'
        if self.f0_extractor == 'parselmouth':
            f0 = parselmouth.Sound(audio, self.sample_rate).to_pitch_ac(
                time_step = self.hop_size / self.sample_rate, 
                voicing_threshold = 0.6,
                pitch_floor = self.f0_min, 
                pitch_ceiling = self.f0_max).selected_array['frequency']
            pad_size = start_frame + (int(len(audio) // self.hop_size) - len(f0) + 1) // 2
            f0 = np.pad(f0,(pad_size, n_frames - len(f0) - pad_size))

        # dio - WORLD's DIO + stonemask by 'pyworld'
        elif self.f0_extractor == 'dio':
            _f0, t = pw.dio(
                audio.astype('double'), 
                self.sample_rate, 
                f0_floor = self.f0_min, 
                f0_ceil = self.f0_max, 
                channels_in_octave=2, 
                frame_period = (1000 * self.hop_size / self.sample_rate))
            f0 = pw.stonemask(audio.astype('double'), _f0, t, self.sample_rate)
            f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))

        # harvest - WORLD's Harvest by 'pyworld'
        elif self.f0_extractor == 'harvest':
            f0, _ = pw.harvest(
                audio.astype('double'), 
                self.sample_rate, 
                f0_floor = self.f0_min, 
                f0_ceil = self.f0_max, 
                frame_period = (1000 * self.hop_size / self.sample_rate))
            f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))

        # crepe - CREPE by 'torchcrepe'
        elif self.f0_extractor == 'crepe':
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            wav16k = resampy.resample(audio, self.sample_rate, 16000)
            wav16k_torch = torch.FloatTensor(wav16k).unsqueeze(0).to(device)

            f0, pd = torchcrepe.predict(wav16k_torch, 16000, 80, self.f0_min, self.f0_max, pad=True, model='full', batch_size=512, device=device, return_periodicity=True)

            pd = torchcrepe.filter.median(pd, 4)
            pd = torchcrepe.threshold.Silence(-60.)(pd, wav16k_torch, 16000, 80)
            f0 = torchcrepe.threshold.At(0.05)(f0, pd)
            f0 = torchcrepe.filter.mean(f0, 4)
            f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)
            
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
        self.hop_size = hop_size
        
    def extract(self, audio): # audio: 1d numpy array
        n_frames = int(len(audio) // self.hop_size) + 1
        audio2 = audio ** 2
        audio2 = np.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode = 'reflect')
        volume = np.array([np.mean(audio2[int(n * self.hop_size) : int((n + 1) * self.hop_size)]) for n in range(n_frames)])
        volume = np.sqrt(volume)
        return volume
    
         
class Units_Encoder:
    def __init__(self, encoder, encoder_ckpt, encoder_sample_rate = 16000, encoder_hop_size = 320, device = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        if encoder == 'hubertsoft':
            self.model = Audio2HubertSoft(encoder_ckpt).to(device)
        else:
            raise ValueError(f" [x] Unknown units encoder: {encoder}")
            
        self.resample_kernel = {}
        self.encoder_sample_rate = encoder_sample_rate
        self.encoder_hop_size = encoder_hop_size
        
    def encode(self, 
                audio, # B, T
                sample_rate,
                hop_size): 
        
        # resample
        if sample_rate == self.encoder_sample_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.encoder_sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
        
        # encode
        if audio_res.size(-1) < self.encoder_hop_size:
            audio_res = torch.nn.functional.pad(audio, (0, self.encoder_hop_size - audio_res.size(-1)))
        units = self.model(audio_res)
        
        # alignment
        n_frames = audio.size(-1) // hop_size + 1
        ratio = (hop_size / sample_rate) / (self.encoder_hop_size / self.encoder_sample_rate)
        index = torch.clamp(torch.round(ratio * torch.arange(n_frames).to(self.device)).long(), max = units.size(1) - 1)
        units_aligned = torch.gather(units, 1, index.unsqueeze(0).unsqueeze(-1).repeat([1, 1, units.size(-1)]))
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
     
    def forward(self, 
                audio): # B, T
        with torch.inference_mode():  
            units = self.hubert.units(audio.unsqueeze(1))
            return units
            
class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__
    
def load_model(
        model_path,
        device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load model
    model = None

    if args.model.type == 'Sins':
        model = Sins(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_harmonics=args.model.n_harmonics,
            n_mag_allpass=args.model.n_mag_allpass,
            n_mag_noise=args.model.n_mag_noise,
            n_unit=args.data.encoder_out_channels,
            n_spk=args.model.n_spk)
    
    elif args.model.type == 'CombSub':
        model = CombSub(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_mag_allpass=args.model.n_mag_allpass,
            n_mag_harmonic=args.model.n_mag_harmonic,
            n_mag_noise=args.model.n_mag_noise,
            n_unit=args.data.encoder_out_channels,
            n_spk=args.model.n_spk)
    
    elif args.model.type == 'CombSubFast':
        model = CombSubFast(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_unit=args.data.encoder_out_channels,
            n_spk=args.model.n_spk)
            
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, args
    
 
class Sins(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_harmonics,
            n_mag_allpass,
            n_mag_noise,
            n_unit=256,
            n_spk=1):
        super().__init__()

        print(' [DDSP Model] Sinusoids Additive Synthesiser')

        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        # Unit2Control
        self.unit2ctrl = Unit2Control(n_unit, n_spk, { 'amplitudes': n_harmonics, 'group_delay': n_mag_allpass, 'noise_magnitude': n_mag_noise, })

    def forward(self, units_frames, f0_frames, volume_frames, spk_id=None, spk_mix_dict=None, initial_phase=None, infer=True, max_upsample_dim=32):
        '''
            units_frames  :: (B, Frame, Feat)
            f0_frames     :: (B, Frame, 1)
            volume_frames :: (B, Frame, 1)
            spk_id        :: (B, 1)
        '''
        # exciter phase
        ## frequency contour
        f0 = upsample(f0_frames, self.block_size)
        _f0 = f0.double() if infer else f0
        ## phase contour, t=0 is 0
        x = torch.cumsum(_f0 / self.sampling_rate, axis=1)
        ## phase contour, init_phase adjusted
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        ## (maybe) phase wrapping
        x = x - torch.round(x)
        x = x.to(f0)
        phase = 2 * np.pi * x
        ## phase at each frame's start
        phase_frames = phase[:, ::self.block_size, :]
        
        # parameter prediction
        ctrls = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk_id=spk_id, spk_mix_dict=spk_mix_dict)
        
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
            phases = phase * level_harmonic[start:end]
            amplitudes = upsample(amplitudes_frames[:,:,start:end], self.block_size)
            # A * sin(phase)
            sinusoids += (amplitudes * torch.sin(phases)).sum(-1)

        # harmonic part filter (apply group-delay)
        harmonic = frequency_filter(sinusoids, torch.exp(1.j * torch.cumsum(group_delay, axis = -1)), hann_window = False)
                        
        # SubNoise 
        noise = torch.rand_like(harmonic) * 2 - 1
        noise = frequency_filter(noise, torch.complex(noise_param, torch.zeros_like(noise_param)), hann_window = True)
                        
        signal = harmonic + noise

        return signal, phase, (harmonic, noise) #, (noise_param, noise_param)


class CombSubFast(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            n_spk=1):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }
        self.unit2ctrl = Unit2Control(n_unit, n_spk, split_map)

    def forward(self, units_frames, f0_frames, volume_frames, spk_id=None, spk_mix_dict=None, initial_phase=None, infer=True, **kwargs):
        '''
            units_frames: B x n_frames x n_unit
            f0_frames: B x n_frames x 1
            volume_frames: B x n_frames x 1 
            spk_id: B x 1
        '''
        # exciter phase
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        ctrls = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk_id=spk_id, spk_mix_dict=spk_mix_dict)
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]

        return signal, phase_frames, (signal, signal)

class CombSub(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_mag_allpass,
            n_mag_harmonic,
            n_mag_noise,
            n_unit=256,
            n_spk=1):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser (Old Version)')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        #Unit2Control
        self.unit2ctrl = Unit2Control(n_unit, n_spk, { 'group_delay': n_mag_allpass, 'harmonic_magnitude': n_mag_harmonic, 'noise_magnitude': n_mag_noise, })

    def forward(self, units_frames, f0_frames, volume_frames, spk_id=None, spk_mix_dict=None, initial_phase=None, infer=True, **kwargs):
        '''
            units_frames  :: (B, Frame, Feat) - Unit series
            f0_frames     :: (B, Frame, 1)    - fo series
            volume_frames :: (B, Frame, 1)    - Volume series
            spk_id        :: (B, 1)           - Speaker ID
            spk_mix_dict
            initial_phase
            infer
        '''
        # exciter phase
        f0 = upsample(f0_frames, self.block_size)
        # 2pi * f0 [rad/s] / sr [sample/s] = [rad/sample]
        _f0 = f0.double() if infer else f0
        x = torch.cumsum(_f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi
        x = x - torch.round(x)
        x = x.to(f0)
        
        # phase at each frame's start
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        ctrls = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk_id=spk_id, spk_mix_dict=spk_mix_dict)
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
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)

        # harmonic part filter (using dynamic-windowed LTV-FIR, with group-delay prediction)
        harmonic = frequency_filter(combtooth, torch.exp(1.j * torch.cumsum(group_delay, axis = -1)), hann_window = False)
        harmonic = frequency_filter(
                        harmonic,
                        torch.complex(src_param, torch.zeros_like(src_param)),
                        hann_window = True,
                        half_width_frames = 1.5 * self.sampling_rate / (f0_frames + 1e-3))

        # noise part filter (using constant-windowed LTV-FIR, without group-delay)
        noise = torch.rand_like(harmonic) * 2 - 1
        noise = frequency_filter(noise, torch.complex(noise_param, torch.zeros_like(noise_param)), hann_window = True)
                        
        signal = harmonic + noise

        return signal, phase_frames, (harmonic, noise)
