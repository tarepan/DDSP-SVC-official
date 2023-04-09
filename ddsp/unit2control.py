import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .pcmer import PCmer


def split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors."""
    labels = []
    sizes = []

    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)

    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))


class Unit2Control(nn.Module):
    def __init__(self, ndim_feat_i, n_spk: int, output_splits, c: bool = False):
        """
        Args:
            input_channel - Feature dimension size of unit series
            n_spk         - The number of speakers
            output_splits - Output shape specifier
        """
        super().__init__()

        ndim_feat = 256
        ndim_out = sum([v for _, v in output_splits.items()])

        # PreNet - Conv-GN-LReLU-Conv
        kernel = 3
        self.prenet = nn.Sequential(
            nn.Conv1d(ndim_feat_i, ndim_feat, kernel, padding="same"),
            nn.GroupNorm(4,        ndim_feat),
            nn.LeakyReLU(),
            nn.Conv1d(ndim_feat,   ndim_feat, kernel, padding="same")
        )

        # Embedding
        ## fo/phase/volume continuous embedding :: (B, Frame, 1) -> (B, Frame, Emb)
        ## spk             discrete   embedding :: (B,)          -> (B, Emb)
        self.f0_embed     =    nn.Linear(1,     ndim_feat)
        self.phase_embed  =    nn.Linear(1,     ndim_feat)
        self.volume_embed =    nn.Linear(1,     ndim_feat)
        self.spk_embed    = nn.Embedding(n_spk, ndim_feat)

        # Conformer decoder & Linear postNet
        num_layers, num_heads = 3, 8
        self.dec_post = nn.Sequential(
            PCmer(num_layers, num_heads, ndim_feat, c),
            nn.LayerNorm(ndim_feat),
            weight_norm(nn.Linear(ndim_feat, ndim_out)),
        )

        # Output split
        self.output_splits = output_splits


    def forward(self, units, f0, phase, volume, spk_id, spk_mix_dict = None):
        """
        Args:
            units  :: (B, Frame, Feat) - Acoustic unit series
            f0     :: (B, 1)           - Fundamental tone's frequency contour
            phase  :: (B, Frame)       - Frame-wise phase  contour (phase at frame start)
            volume :: (B, Frame)       - Frame-wise volume contour (non-overlapped RMS of the waveform)
            spk_id :: (B,)             - Speaker index
            spk_mix_dict
        return: 
            dict of (B, Frame, Feat)   - Feature serieses
        """

        # PreNet :: (B, Frame, Feat) -> (B, Feat, Frame) -> (B, Feat, Frame) -> (B, Frame, Feat)
        x = self.prenet(units.transpose(1,2)).transpose(1,2)

        # Embedding
        ## Add continuous embeddings of fo/phase/volume to processed unit
        phase, volume = phase.unsqueeze(-1), volume.unsqueeze(-1)
        x = x + self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume)
        ## Add discrete or mixed discrete embeddings of spk to others
        if spk_mix_dict is not None:
            # Speaker mixing - weighted sum of each speaker's embeddings
            for k, v in spk_mix_dict.items():
                spk_id_tensor = torch.LongTensor(np.array([[k]])).to(units.device)
                x = x + v * self.spk_embed(spk_id_tensor - 1)
        else:
            # Speaker embedding
            x = x + self.spk_embed(spk_id - 1)

        # Conformer/PostNet
        e = self.dec_post(x)

        return split_to_dict(e, self.output_splits) 
