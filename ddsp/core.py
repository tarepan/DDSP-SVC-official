import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


def upsample(signal, factor):
    """
    Args:
        signal (at least in CombSubFast, :: (B, Frame, 1))
        factor :: - Scale factor
    """
    # (B, Frame, Feat) -> (B, Feat, Frame)
    signal = signal.permute(0, 2, 1)

    # (B, Feat, Frame) & (B, Feat, 1) -> (B, Feat, Frame+1) -> (B, Feat, factor*Frame+1) -> (B, Feat, factor*Frame)
    signal = nn.functional.interpolate(torch.cat((signal,signal[:,:,-1:]),2), size=signal.shape[-1] * factor + 1, mode='linear', align_corners=True)
    signal = signal[:,:,:-1]

    signal = signal.permute(0, 2, 1)
    return signal


def remove_above_fmax(amplitudes, pitch, fmax, level_start=1):
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(level_start, n_harm + level_start).to(pitch)
    aa = (pitches < fmax).float() + 1e-7
    return amplitudes * aa


def fo_to_rot(fo, sr: int, initial_phase = None, precise: bool = False):
    """
    Args:
        fo            :: (B, T) - Instantaneous frequency series [Hz]
        initial_phase :: (B,)   - Initial phase [rad]
    Returns:
        rot           :: (B, T) - Wrapped rotation [・], within range (-0.5, 0.5]
    """
    # Convert to fp64 for smaller numerial error
    _fo = fo.double() if precise else fo

    # fo to rotation by norm + cumsum + init + wrapping
    rot = torch.cumsum(_fo / sr, axis=1)
    if initial_phase is not None:
        rot += initial_phase.unsqueeze(-1).to(rot) / 2 / np.pi
    rot = rot - torch.round(rot)

    # Back from fp64
    rot = rot.to(fo)

    return rot


def test_fo_to_rot_dtype():
    fo = torch.tensor([[1.0, 1.0, 1.0,]])
    rot_np = fo_to_rot(fo, 1, precise=False)
    rot_p  = fo_to_rot(fo, 1, precise=True)
    assert fo.dtype == rot_np.dtype, f"{fo.dtype} != {rot_np.dtype}"
    assert fo.dtype == rot_p.dtype, f"{fo.dtype} != {rot_p.dtype}"


def test_fo_to_rot_stablefo():
    fo_contour = torch.tensor([[ 1.0,   1.0,   1.0, ]])
    sr = 4
    rot_gt     = torch.tensor([[+0.25, +0.50, -0.25,]])
    rot_calc   = fo_to_rot(fo_contour, sr, initial_phase=None, precise=False)
    assert torch.allclose(rot_gt, rot_calc), f"{rot_gt} != {rot_calc}"
    

def test_fo_to_rot_fm():
    fo_contour = torch.tensor([[ 1.0,   2.0,   3.0, ]])
    sr = 4
    # rot_gt   = torch.tensor([[+0.25, +0.75, +1.50,]])
    rot_gt     = torch.tensor([[+0.25, -0.25, -0.50,]])
    rot_calc   = fo_to_rot(fo_contour, sr, initial_phase=None, precise=False)
    assert torch.allclose(rot_gt, rot_calc), f"{rot_gt} != {rot_calc}"


def test_fo_to_rot_init_phase():
    import math
    fo_contour = torch.tensor([[ 1.0,   1.0,   1.0, ]])
    init_phase = torch.tensor([1.0 * math.pi,])
    sr = 4
    # rot_gt   = torch.tensor([[+0.75, +1.00, +1.25,]])
    rot_gt     = torch.tensor([[-0.25, +0.00, +0.25,]])
    rot_calc   = fo_to_rot(fo_contour, sr, initial_phase=init_phase, precise=False)
    assert torch.allclose(rot_gt, rot_calc), f"{rot_gt} != {rot_calc}"


def test_fo_to_rot_fm_init_batch():
    import math
    fo_contour = torch.tensor([[ 1.0,   1.0,   1.0, ], [ 1.0,   2.0,   3.0, ],])
    init_phase = torch.tensor([  1.0 * math.pi       ,   0.0                 ,])
    sr = 4
    rot_gt     = torch.tensor([[-0.25, +0.00, +0.25,], [+0.25, -0.25, -0.50,],])
    rot_calc   = fo_to_rot(fo_contour, sr, initial_phase=init_phase, precise=True)
    assert torch.allclose(rot_gt, rot_calc, atol=1e-05), f"{rot_gt} != {rot_calc}"

def MaskedAvgPool1d(x, kernel_size):
    x = x.unsqueeze(1)
    x = F.pad(x, ((kernel_size - 1) // 2, kernel_size // 2), mode="reflect")
    mask = ~torch.isnan(x)
    masked_x = torch.where(mask, x, torch.zeros_like(x))
    ones_kernel = torch.ones(x.size(1), 1, kernel_size, device=x.device)

    # Perform sum pooling
    sum_pooled = F.conv1d(
        masked_x,
        ones_kernel,
        stride=1,
        padding=0,
        groups=x.size(1),
    )

    # Count the non-masked (valid) elements in each pooling window
    valid_count = F.conv1d(
        mask.float(),
        ones_kernel,
        stride=1,
        padding=0,
        groups=x.size(1),
    )
    valid_count = valid_count.clamp(min=1)  # Avoid division by zero

    # Perform masked average pooling
    avg_pooled = sum_pooled / valid_count

    return avg_pooled.squeeze(1)

def MedianPool1d(x, kernel_size):
    x = x.unsqueeze(1)
    x = F.pad(x, ((kernel_size - 1) // 2, kernel_size // 2), mode="reflect")
    x = x.squeeze(1)
    x = x.unfold(1, kernel_size, 1)
    x, _ = torch.sort(x, dim=-1)
    return x[:, :, (kernel_size - 1) // 2]
    
#### Filter #######################################################################################
def _get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True):
  """Calculate final size for efficient FFT.
  Args:
    frame_size: Size of the audio frame.
    ir_size: Size of the convolving impulse response.
    power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
      numbers. TPU requires power of 2, while GPU is more flexible.
  Returns:
    fft_size: Size for efficient FFT.
  """
  convolved_frame_size = ir_size + frame_size - 1
  if power_of_2:
    # Next power of 2.
    fft_size = int(2**np.ceil(np.log2(convolved_frame_size)))
  else:
    fft_size = convolved_frame_size
  return fft_size


def _crop_and_compensate_delay(audio, audio_size, ir_size, padding: str = 'same', delay_compensation = -1):
    """Crop audio output from convolution to compensate for group delay.
    Args:
        audio :: (B, T)             - Audio after convolution
        audio_size                  - Initial size of the audio before convolution.
        ir_size                     - Size of the convolving impulse response.
        padding :: 'valid' | 'same'
            'same'  - the final output to be the same size as the input audio
            'valid' - the audio is extended to include the tail of the impulse response (audio_timesteps + ir_timesteps - 1)
        delay_compensation          - Size of head trimming as group delay compensation [sample].
                                        If <0, automatically calculating a constant group delay of the windowed linear phase filter from _frequency_impulse_response().
    Returns:
        Tensor of cropped and shifted audio.
    """
    # NOTE: Current usage is padding='same' & delay_compensation=-1

    # Crop the output.
    if padding == 'valid':
        crop_size = ir_size + audio_size - 1
    elif padding == 'same':
        crop_size = audio_size

    # total_size = T
    total_size = int(audio.shape[-1])
    # crop = T - audio_size
    crop = total_size - crop_size

    # For an impulse response produced by _frequency_impulse_response(), the group delay is constant because the filter is linear phase.
    # Head trimming as delay compensation
    # start = ir_size // 2
    start = (ir_size // 2 if delay_compensation < 0 else delay_compensation)
    # end = crop - start = (T - audio_size) - (ir_size // 2)
    end = crop - start

    # :: (B, T) -> (B, T')
    return audio[:, start:-end]


def _fft_convolve(audio, impulse_response):
    """Apply LTV-FIR filter through frequency domain over frames.

    1. splits the audio into frames based on the number of IR frames
    2. applies filters through frequency domain
    3. overlap-and-adds audio back together
    4. compensate group delay 

    Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute convolution for large impulse response sizes.

    Args:
        audio            :: (B, T)                   - Input audio
        impulse_response :: (B, Frame, 2*(n_mags-1)) - Finite impulse response to convolve.
                            (B, ir_size) | (B, Frame, ir_size) - Finite impulse response of LTI or LTV filter. Automatically chops the audio into equally shaped blocks to match ir_frames.
    Returns:
        audio_out        :: (B, T)                   - Filtered audio
    """
    # Shape
    ## Unsqueeze :: (B, ir_size) | (B, Frame, ir_size) -> (B, Frame=1, ir_size) | (B, Frame, ir_size)
    ir_shape = impulse_response.size() 
    if len(ir_shape) == 2:
        impulse_response = impulse_response.unsqueeze(1)
        ir_shape = impulse_response.size()
    ## Get shapes
    batch_size_ir, n_ir_frames, ir_size = ir_shape
    batch_size, audio_size = audio.size() # B, T
    # Validation
    if batch_size != batch_size_ir:
        raise ValueError(f'Batch size of audio ({batch_size}) and impulse response ({batch_size_ir}) must be the same.')

    # Cut audio into 50% overlapped frames (center padding).
    hop_size = int(audio_size / n_ir_frames)
    frame_size = 2 * hop_size    
    audio_frames = F.pad(audio, (hop_size, hop_size)).unfold(1, frame_size, hop_size)
    
    # Apply Bartlett (triangular) window
    window = torch.bartlett_window(frame_size).to(audio_frames)
    audio_frames = audio_frames * window

    # Filtering (convolution) - through frequency domain by FFT/Multiply/iFFT
    ## Pad and FFT the audio and impulse responses.
    fft_size = _get_fft_size(frame_size, ir_size, power_of_2=False)
    audio_fft = torch.fft.rfft(audio_frames,                                              fft_size)
    ir_fft    = torch.fft.rfft(torch.cat((impulse_response,impulse_response[:,-1:,:]),1), fft_size)
    audio_ir_fft = torch.multiply(audio_fft, ir_fft)
    audio_frames_out = torch.fft.irfft(audio_ir_fft, fft_size)
    
    # Overlap Add
    batch_size, n_audio_frames, frame_size = audio_frames_out.size() # # B, n_frames+1, 2*(hop_size+n_mags-1)-1
    fold = torch.nn.Fold(output_size=(1, (n_audio_frames - 1) * hop_size + frame_size),kernel_size=(1, frame_size),stride=(1, hop_size))
    output_signal = fold(audio_frames_out.transpose(1, 2)).squeeze(1).squeeze(1)
    
    # Crop and shift the output audio. ? & Delay compensation
    output_signal = _crop_and_compensate_delay(output_signal[:,hop_size:], audio_size, ir_size)
    return output_signal


def _apply_window_to_impulse_response(impulse_response, window_size: int = 0, causal: bool = False):
    """Apply a window to an impulse response and put in causal form.
    Args:
        impulse_response :: (B, Frame, ir_size=2*(n_mag-1)) - A series of impulse responses frames to window, of shape
          ---------> ir_size means size of filter_bank ??????
        window_size                             - Size of the window to apply in the time domain. If <1, it defaults to the impulse_response size.
        causal                                  - Whether `impulse_response` is in causal form (peak in the middle) or not
    Returns:
        impulse_response: Windowed IR in causal form, with last dimension cropped to window_size if window_size is greater than 0 and less than ir_size.
    """

    # If IR is in causal form, put it in zero-phase form.
    if causal:
        impulse_response = torch.fftshift(impulse_response, axes=-1)

    # Get a window for better time/frequency resolution than rectangular.
    # Window defaults to IR size, cannot be bigger.
    ir_size = int(impulse_response.size(-1))
    if (window_size <= 0) or (window_size > ir_size):
        window_size = ir_size
    window = nn.Parameter(torch.hann_window(window_size), requires_grad = False).to(impulse_response)
    
    # Zero pad the window and put in in zero-phase form.
    padding = ir_size - window_size
    if padding > 0:
        half_idx = (window_size + 1) // 2
        window = torch.cat([window[half_idx:],
                            torch.zeros([padding]),
                            window[:half_idx]], axis=0)
    else:
        window = window.roll(window.size(-1)//2, -1)

    # Apply the window, to get new IR (both in zero-phase form).
    window = window.unsqueeze(0)
    impulse_response = impulse_response * window

    # Put IR in causal form and trim zero padding.
    if padding > 0:
        first_half_start = (ir_size - (half_idx - 1)) + 1
        second_half_end = half_idx + 1
        impulse_response = torch.cat([
                                      impulse_response[..., first_half_start : ],
                                      impulse_response[...,   : second_half_end],
                                     ], dim=-1)
    else:
        impulse_response = impulse_response.roll(impulse_response.size(-1)//2, -1)

    return impulse_response


def _apply_dynamic_window_to_impulse_response(impulse_response,  # B, n_frames, 2*(n_mag-1) or 2*n_mag-1
                                             half_width_frames):        # B，n_frames, 1
    ir_size = int(impulse_response.size(-1)) # 2*(n_mag -1) or 2*n_mag-1
    
    window = torch.arange(-(ir_size // 2), (ir_size + 1) // 2).to(impulse_response) / half_width_frames 
    window[window > 1] = 0
    window = (1 + torch.cos(np.pi * window)) / 2 # B, n_frames, 2*(n_mag -1) or 2*n_mag-1
    
    impulse_response = impulse_response.roll(ir_size // 2, -1)
    impulse_response = impulse_response * window
    
    return impulse_response
    
        
def _frequency_impulse_response(magnitudes, hann_window = True, half_width_frames = None):
    """
    no-window -> iFFT + roll
    window ----> iFFT

    Args:
        magnitudes
    """

    # Frequency response to Implulse response
    impulse_response = torch.fft.irfft(magnitudes) # (B, Frame, 2*(n_mags-1))
    
    # Window and put in causal form.
    if hann_window:
        if half_width_frames is None:
            impulse_response = _apply_window_to_impulse_response(impulse_response)
        else:
            impulse_response = _apply_dynamic_window_to_impulse_response(impulse_response, half_width_frames)
    else:
        # Shift last dimension
        impulse_response = impulse_response.roll(impulse_response.size(-1) // 2, -1)
       
    return impulse_response


def frequency_filter(audio, magnitudes, hann_window=True, half_width_frames=None):
    """Apply filter toward an audio."""
    # Frequency response to linear-phase LTV-FIR
    impulse_response = _frequency_impulse_response(magnitudes, hann_window, half_width_frames)
    # Apply linear-phase LTV-FIR filter through frequency domain
    return _fft_convolve(audio, impulse_response)
#### /Filter ######################################################################################
