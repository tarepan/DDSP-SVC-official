<div align="center">

# DDSP-SVC <!-- omit in toc -->
[![ColabBadge]][notebook]

</div>

Clone of singing voice conversion based on DDSP.

<!-- Auto-generated by "Markdown All in One" extension -->
- [Demo](#demo)
- [Usage](#usage)
  - [Install](#install)
  - [Train](#train)
  - [Inference](#inference)
- [Results](#results)
- [References](#references)

## Demo
Samples or Link to [demo page].  

## Introduction
Features compared to [Diff-SVC](https://github.com/prophesier/diff-svc) and [SO-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc):

- light training
- fast inference
- Equivalent or lower quality
  - sub-optimal quality w/o enhancer
  - optimal quality w/ enhancer (close to SO-VITS-SVC, not the level of Diff-SVC)

## Usage
### Install

```bash
# pip install "torch==1.11.0" -q      # Based on your environment (validated with 1.9.1)
# pip install "torchaudio==0.11.0" -q # Based on your environment (validated with 0.6.0)
# pip install git+https://github.com/tarepan/DDSP-SVC-official
```

```bash
git clone https://github.com/tarepan/DDSP-SVC-official
```

```bash
pip install -r requirements.txt 
```

Then, configure the pretrained models:

- **(Required)** Prepare encoder
  - TypeA: Download the pretrained [**HubertSoft**](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt) encoder and put it under `pretrain/hubert` folder.
  - TypeB: Download the pretrained [**ContentVec**](https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
-  Get the pretrained vocoder-based enhancer from the [DiffSinger Community Vocoders Project](https://openvpi.github.io/vocoders) and unzip it into `pretrain/` folder

### Preprocessing

First, place audio files under the directory structured like below:
```bash
data/
    val/audio/
    train/audio/
        1/
        2/
            ccc.wav
            ddd.wav
```

Speaker folder name should be **positive integers not greater than 'n_spk'** to represent speaker ids.  

You can also run
```bash
python draw.py
```
to help you select validation data (you can adjust the parameters in `draw.py` to modify the number of extracted files and other parameters)

Then run:
```bash
python preprocess.py -c <seletct_config_as___configs/combsub.yaml>
```

- Configs
  - sampling rate: 44.1khz
  - commons: optimized for NVIDIA GTX 1660
- Restrictions
  - assert 'audio_sr == config_sr'. If not, training becomes very slow by resampling.
  - assert '2sec <= len(audio) <= not_too_long'
  - assert 'n_clip ~ 1000 if cache_all_data is True' because of on-memory cache size
  - assert 'n_val_clip <= 10' because of validation cost

### Train
Jump to ☞ [![ColabBadge]][notebook], then Run. That's all!  

For arguments, check [./ddspsvc/config.py](https://github.com/user_name/DDSP-SVC-official/blob/main/ddspsvc/config.py).  
For dataset, check [`speechcorpusy`](https://github.com/tarepan/speechcorpusy).  

```bash
python train.py -c <seletct_config_as___configs/combsub.yaml>
```
The test audio samples in Tensorboard are the outputs of DDSP-SVC w/o enhancer.  

You can safely interrupt training, then running the same command line will resume training.

You can also finetune the model if you interrupt training first, then re-preprocess the new dataset or change the training parameters (batchsize, lr etc.) and then run the same command line.

### Inference
Both CLI and Python supported.  
For detail, jump to ☞ [![ColabBadge]][notebook] and check it.  

Pretrained model is provided in GitHub release.  
With provided model or your trained model, run:
```bash
# Pure DDSP
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -id <speaker_id> -e false

# PPSP + enhancer
## if normal vocal range, set `enhancer_adaptive_key` to 0, else to >0
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -id <speaker_id> -eak <enhancer_adaptive_key (semitones)>
```

```bash
python main.py -h
```

UPDATE：Mix-speaker is supported now. You can use "-mix" option to design your own vocal timbre, below is an example:
```bash
# Mix the timbre of 1st and 2nd speaker in a 0.5 to 0.5 ratio
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -mix "{1:0.5, 2:0.5}" -eak 0
```

### Real-time VC
Start a simple GUI with the following command:
```bash
python gui.py
```
The front-end uses technologies such as sliding window, cross-fading, SOLA-based splicing and contextual semantic reference, which can achieve sound quality close to non-real-time synthesis with low latency and resource occupation.

## Results
### Sample <!-- omit in toc -->
[Demo](#demo)

### Performance <!-- omit in toc -->
- training
  - x.x [iter/sec] @ NVIDIA X0 on Google Colaboratory (AMP+)
  - take about y days for whole training
- inference
  - z.z [sec/sample] @ xx

## References
### Origin <!-- omit in toc -->

### Acknowlegements <!-- omit in toc -->
- [ddsp](https://github.com/magenta/ddsp)
- [pc-ddsp](https://github.com/yxlllc/pc-ddsp)
- [soft-vc](https://github.com/bshall/soft-vc)
- [DiffSinger (OpenVPI version)](https://github.com/openvpi/DiffSinger)


[ColabBadge]:https://colab.research.google.com/assets/colab-badge.svg

[notebook]:https://colab.research.google.com/github/tarepan/DDSP-SVC-official/blob/main/ddspsvc.ipynb
[demo page]:https://demo.project.your
