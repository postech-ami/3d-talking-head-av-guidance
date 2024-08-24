import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
import re

from auto_avsr.preparation.transforms import TextTransform


class Dataset(data.Dataset):
    def __init__(self, data, subjects_dict, data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))

    def __getitem__(self, index):
        if self.data_type == "train":
            file_name = self.data[index]["name"]
            subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels
            
        return_dict = {
            "audio": torch.FloatTensor(self.data[index]["audio"]),
            "vertice": torch.FloatTensor(self.data[index]["vertice"]),
            "template": torch.FloatTensor(self.data[index]["template"]), 
            "one_hot": torch.FloatTensor(one_hot),
            "file_name": file_name,
            "sentence": self.data[index]["sentence"],
            "text_token": torch.Tensor(self.data[index]["text_token"]),
            "waveform": torch.FloatTensor(self.data[index]["waveform"]),
        }
        return return_dict


    def __len__(self):
        return self.len


def read_data(args):
    data = defaultdict(dict)

    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.dataset, args.wav_path)
    vertices_path = os.path.join(args.dataset, args.vertices_path)

    # Audio processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    # Templates (dict): key = identity, value = ndarray with (num_verts, 3)
    template_file = os.path.join(args.dataset, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    # Transcripts
    sentence_path = os.path.join(args.dataset, args.sentence_path)
    sentences = {}
    if args.dataset == "vocaset":
        for sentence_file in os.listdir(sentence_path):
            if not sentence_file.endswith("txt"):
                continue
            identity = sentence_file[:sentence_file.find('.txt')]
            sentence_file = os.path.join(sentence_path, sentence_file)
            with open(sentence_file) as f:
                lines = f.readlines()
                sentences[identity] = [line.strip() for line in lines if line!="\n"]
    elif args.dataset=="BIWI":
        with open(sentence_path) as f:
            lines = f.readlines()
        for l_idx, l in enumerate(lines):
            strt_idx = l.find(". ")
            sentences[l_idx+1] = l.strip()[strt_idx+2:]
    
    # Text transform
    texttransform = TextTransform()
    
    # Read data
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs, desc="Loading data"):
            # Check file existance
            if not f.endswith("wav"):
                continue
            vertice_path = os.path.join(vertices_path, f.replace("wav", "npy"))
            if not os.path.exists(vertice_path):
                continue

            key = f.replace("wav", "npy")
            subject_id = "_".join(key.split("_")[:-1])
            data[key]["name"] = f
            
            # Audio
            wav_path = os.path.join(r,f)
            speech_array, _ = librosa.load(wav_path, sr=16000)
            input_values = np.squeeze(
                processor(speech_array, sampling_rate=16000).input_values
            )
            data[key]["audio"] = input_values # np.ndarray, shape - (len_signal,)
            data[key]["waveform"] = speech_array # np.ndarray, shape - (len_signal,)
            
            # Template (ndarray): shape - (num_verts*3)
            temp = templates[subject_id]
            data[key]["template"] = temp.reshape((-1)) 

            # GT mesh (ndarray): shape - (frames, num_verts*3)
            if args.dataset=="vocaset": # due to the memory limit
                data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:]
            elif args.dataset=="BIWI":
                data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)

            # GT sentences & Text tokens
            if args.dataset=="vocaset":
                identity = key[:key.find("_sentence")]
                s_idx_end = f.find('.wav')
                sentence_num = int(f[s_idx_end-2:s_idx_end])-1 # list starts from 0
                text = sentences[identity][sentence_num]
            elif args.dataset=="BIWI":
                s_idx_end = f.find('.wav')
                sentence_num = int(f[s_idx_end-2:s_idx_end])
                text = sentences[sentence_num]
            text = re.sub(r"[^a-zA-Z]", " ", text).strip()
            token_ids = texttransform.tokenize(text) # torch.tensor
            data[key]["sentence"] = text
            data[key]["text_token"] = token_ids.tolist()

    # Split dataset
    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]

    splits = {
        'vocaset':{'train':range(1, 41), 'val':range(21, 41), 'test':range(21, 41)},
        'BIWI':{'train': range(1, 33),'val': range(33, 37),'test': range(37, 41)},
    }
   
    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-1])
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects_dict["train"] and sentence_id in splits[args.dataset]['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits[args.dataset]['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits[args.dataset]['test']:
            test_data.append(v)

    print(f"Train: {len(train_data)}, Val: {len(valid_data)}, Test: {len(test_data)}")
    return train_data, valid_data, test_data, subjects_dict


def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data, subjects_dict, "train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    valid_data = Dataset(valid_data, subjects_dict, "val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = Dataset(test_data, subjects_dict, "test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return dataset


if __name__ == "__main__":
    get_dataloaders()
    