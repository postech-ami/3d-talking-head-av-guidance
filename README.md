# Enhancing Speech-Driven 3D Facial Animation with Audio-Visual Guidance from Lip Reading Expert (INTERSPEECH 24)

### [Project Page](https://3d-talking-head-avguide.github.io/) | [Paper](https://arxiv.org/abs/2407.01034)

This repository contains the official implementation of the INTERSPEECH 2024 paper, "Enhancing Speech-Driven 3D Facial Animation with Audio-Visual Guidance from Lip Reading Expert"

## Getting Started

### Installation

This code was developed on Ubuntu 18.04 with Python 3.8, CUDA 11.3, and Pytorch 1.10.0.

Clone this repo:

```
git clone https://github.com/postech-ami/3d-talking-head-av-guidance
cd 3d-talking-head-av-guidance
```

Make a virtual environment:

```
conda create --name av_guidance python=3.8 -y
conda activate av_guidance
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install pytorch-lightning==1.5.10
pip install hydra-core --upgrade
conda install -c conda-forge ffmpeg
pip install -r requirements.txt 
```

Compile and install psbody-mesh package: [MPI-IS/mesh](https://github.com/MPI-IS/mesh):
```
BOOST_INCLUDE_DIRS=/usr/lib/x86_64-linux-gnu make all
```

### Lip reading expert

For your convenience, download the model weight [here](https://drive.google.com/file/d/1mU6MHzXMiq1m6GI-8gqT2zc2bdStuBXu/view?usp=sharing), and fill in the configuration `lipreader_path` with the path of model.

Clone the repository [Auto-AVSR](https://github.com/mpc001/auto_avsr) in this directory and update the import lines of the below files as `auto_avsr/[existing_imports]`. For instance,

```py
# BEFORE
from espnet.nets.pytorch_backend.e2e_asr_conformer_av import E2E
# AFTER
from auto_avsr.espnet.nets.pytorch_backend.e2e_asr_conformer_av import E2E
```


<details>
<summary>List of files</summary>

```
espnet/nets/pytorch_backend/backbones/modules/resnet.py
espnet/nets/pytorch_backend/backbones/modules/resnet1d.py

espnet/nets/pytorch_backend/backbones/conv1d_extractor.py
espnet/nets/pytorch_backend/backbones/conv3d_extractor.py

espnet/nets/pytorch_backend/transformer/add_sos_eos.py
espnet/nets/pytorch_backend/transformer/decoder.py
espnet/nets/pytorch_backend/transformer/decoder_layer.py
espnet/nets/pytorch_backend/transformer/encoder_layer.py
espnet/nets/pytorch_backend/transformer/encoder.py

espnet/nets/pytorch_backend/ctc.py
espnet/nets/pytorch_backend/e2e_asr_conformer_av.py
espnet/nets/pytorch_backend/e2e_asr_conformer.py ??
espnet/nets/pytorch_backend/nets_utils.py

espnet/nets/scorers/ctc.py
espnet/nets/scorers/length_bonus.py

espnet/nets/batch_beam_search.py
espnet/nets/beam_search.py

lightning_av.py
```
</details>




## Datasets

### VOCASET

Request the VOCASET data from https://voca.is.tue.mpg.de/. Place the downloaded files `data_verts.npy`, `raw_audio_fixed.pkl`, `templates.pkl` and `subj_seq_to_idx.pkl` in the folder `vocaset/`. Download "FLAME_sample.ply" from [VOCA](https://github.com/TimoBolkart/voca/tree/master/template) and put it in `vocaset/`. Read the vertices/audio data and convert them to .npy/.wav files stored in `vocaset/vertices_npy` and `vocaset/wav` folder using a [script](https://github.com/EvelynFan/FaceFormer/blob/main/vocaset/process_voca_data.py).

Download FLAME model and fill the configuration `obj_filename` in `config/vocaset.yaml` with the path of `head_template.obj`.


### BIWI

Follow the [instructions of CodeTalker](https://github.com/Doubiiu/CodeTalker/blob/main/BIWI/README.md) to preprocess BIWI dataset and put .npy/.wav files into `BIWI/vertices_npy` and `BIWI/wav`, and the `templates.pkl` into `BIWI/` folder.

To get the vertex indices of lip geion, download [indices list](https://github.com/Doubiiu/CodeTalker/blob/main/BIWI/regions/lve.txt) and locate it at `BIWI/lve.txt`.

> 2024.08.24. | Unfortunately, BIWI dataset is not available now.



## Training and Testing on VOCASET

- To train the model on VOCASET, run:

```bash
python main.py --dataset vocaset
```

The trained models will be saved to `outputs/model`.

- To test the model on VOCASET, run:

```bash
python test.py --dataset vocaset --test_model_path [path_of_model_weight]
```

The results will be saved to `outputs/pred`. You can download the pretrained model from [faceformer_avguidance_vocaset.pth](https://drive.google.com/file/d/1qVu8zzHjatbyPCIPrWZS6jB4I8kiA2rw/view?usp=sharing).

- To visualize the results, run:
```bash
python render.py --dataset vocaset
```

The results will be saved to `outputs/video`.


## Training and Testing on BIWI

- To train the model on BIWI, run:

```bash
python main.py --dataset BIWI
```

The trained models will be saved to `outputs/model`

- To test the model on BIWI, run:

```bash
python test.py --dataset BIWI --test_model_path [path_of_model_weight]
```

The results will be saved to `outputs/pred`. You can download the pretrained model from [faceformer_avguidance_biwi.pth](https://drive.google.com/file/d/1HpY43-Rw4gbQOB9-4X75hOI5L7O98X3J/view?usp=sharing).

- To visualize the results, run:
```bash
python render.py --dataset BIWI
```

The results will be saved to `outputs/video`.


## Citation

If you find this code useful for your work, please consider citing:

```
@inproceeding{eungi2024enhancing,
	title={Enhancing Speech-Driven 3D Facial Animation with Audio-Visual Guidance from Lip Reading Expert},
	author={EunGi, Han and Hyun-Bin, Oh and Sung-Bin, Kim and Nivelet Etcheberry, Corentin and Nam, Suekyeong and Ju, Janghoon and Oh, Tae-Hyun},
	booktitle={INTERSPEECH 2024},
	year={2024}
}
```

## Acknowledgement

We heavily borrow the code from [FaceFormer](https://github.com/EvelynFan/FaceFormer). We sincerely appreciate those authors.
