# DDSP-SVC
Clone of singing voice conversion based on DDSP.

## Introduction
Features compared to [Diff-SVC](https://github.com/prophesier/diff-svc) and [SO-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc):

- light training
- fast inference
- Equivalent or lower quality
  - sub-optimal quality w/o enhancer
  - optimal quality w/ enhancer (close to SO-VITS-SVC, not the level of Diff-SVC)

## Setup
```bash
pip install -r requirements.txt 
```
Tested on python 3.8 (windows) + pytorch 1.9.1 + torchaudio 0.6.0

Then, configure the pretrained models:

- **(Required)** Download the pretrained [**HubertSoft**](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)   encoder and put it under `pretrain/hubert` folder.
-  Get the pretrained vocoder-based enhancer from the [DiffSinger Community Vocoders Project](https://openvpi.github.io/vocoders) and unzip it into `pretrain/` folder

## Preprocessing

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
  - 'n_spk'
    - `1`: single-speaker model
    - `2` or more: multi-speaker model
- Restrictions
  - assert 'audio_sr == config_sr'. If not, training becomes very slow by resampling.
  - assert '2sec <= len(audio) <= not_too_long'
  - assert 'n_clip ~ 1000 if cache_all_data is True' because of on-memory cache size
  - assert 'n_val_clip <= 10' because of validation cost

## Training
```bash
python train.py -c <seletct_config_as___configs/combsub.yaml>
```
The test audio samples in Tensorboard are the outputs of DDSP-SVC w/o enhancer.  

You can safely interrupt training, then running the same command line will resume training.

You can also finetune the model if you interrupt training first, then re-preprocess the new dataset or change the training parameters (batchsize, lr etc.) and then run the same command line.

## Inference
Pretrained model is provided in GitHub release.  
With provided model or your trained model, run:
```bash
# Pure DDSP
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -id <speaker_id>

# PPSP + enhancer
## if normal vocal range, set `enhancer_adaptive_key` to 0, else to >0
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -id <speaker_id> -e true -eak <enhancer_adaptive_key (semitones)>
```

```bash
# other options about the f0 extractor and response threhold, see
python main.py -h
```

UPDATE：Mix-speaker is supported now. You can use "-mix" option to design your own vocal timbre, below is an example:
```bash
# Mix the timbre of 1st and 2nd speaker in a 0.5 to 0.5 ratio
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -mix "{1:0.5, 2:0.5}" -e true -eak 0

```
## HTTP Server and VST supported
Start the server with the following command
```bash
# configs are in this python file, see the comments (Chinese only)
python flask_api.py
```
Currently supported VST client:
https://github.com/zhaohui8969/VST_NetProcess-

## Acknowledgement
- [ddsp](https://github.com/magenta/ddsp)
- [pc-ddsp](https://github.com/yxlllc/pc-ddsp)
- [soft-vc](https://github.com/bshall/soft-vc)
- [DiffSinger (OpenVPI version)](https://github.com/openvpi/DiffSinger)
