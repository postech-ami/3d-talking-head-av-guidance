import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torch

from data.data_loader import get_dataloaders
from models.base import BaseModel
from utils.utils import *


@torch.no_grad()
def get_predictions(args, model, test_loader):
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    for data in tqdm(test_loader):
        audio = data["audio"].cuda()
        template = data["template"].cuda()
        one_hot_all = data["one_hot"].cuda()
        file_name = data["file_name"]

        train_subject = "_".join(file_name[0].split("_")[:-1])

        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:,iter,:]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze() # (frame, num_verts*3)
            np.save(
                os.path.join(
                    args.save_pred_path, 
                    file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"
                ), 
                prediction.detach().cpu().numpy()
            )
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:,iter,:]
                prediction = model.predict(audio, template, one_hot)
                prediction = prediction.squeeze() # (frame, num_verts*3)
                np.save(
                    os.path.join(
                        args.save_pred_path, 
                        file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"
                    ), 
                    prediction.detach().cpu().numpy()
                )


def main():
    assert torch.cuda.is_available(), "You need cuda to train"

    # Load configurations
    parser = argparse.ArgumentParser(
        description="Speech-Driven 3D Facial Animation with A-V Guidance"
    )
    parser.add_argument(
        "--dataset", type=str, default="vocaset", help="Dataset to train the model",
    )
    parser.add_argument(
        "--test_model_path",
        type=str,
        default="",
        help="Path to model weight file"
    )
    cmd_input = parser.parse_args()
    if cmd_input.dataset=="vocaset":
        args = load_config("config/vocaset.yaml")
    elif cmd_input.dataset == "BIWI":
        args = load_config("config/biwi.yaml")
    if os.path.exists(args.test_model_path):
        setattr(args, "test_model_path", cmd_input.test_model_path)
    
    # Make directories to save results
    make_dirs(args.save_pred_path)

    # Build model - facial animator & lip reader
    model = BaseModel(args, mode="test")
    model.load_model()
    model.eval_mode()

    # Load data
    dataset = get_dataloaders(args)

    # Test the model
    get_predictions(args, model.facial_animator, dataset["test"])

if __name__=="__main__":
    main()