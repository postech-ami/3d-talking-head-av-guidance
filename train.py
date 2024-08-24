import argparse
import warnings
warnings.filterwarnings("ignore")
import torch

from data.data_loader import get_dataloaders
from models.base import BaseModel
from run.trainer import trainer
from utils.utils import *


def main():
    assert torch.cuda.is_available(), "You need cuda to train"

    # Load configurations
    parser = argparse.ArgumentParser(
        description="Speech-Driven 3D Facial Animation with A-V Guidance"
    )
    parser.add_argument(
        "--dataset", type=str, default="vocaset", help="Dataset to train the model",
    )
    args = parser.parse_args()
    if args.dataset=="vocaset":
        args = load_config("config/vocaset.yaml")
    elif args.dataset == "BIWI":
        args = load_config("config/biwi.yaml")

    # Make directories to save results
    make_dirs(args.save_model_path)

    # Build model - facial animator & lip reader
    model = BaseModel(args)

    # Load data
    dataset = get_dataloaders(args)    

    # Train the model
    trainer(args, dataset, model)


if __name__=="__main__":
    main()