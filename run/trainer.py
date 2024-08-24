import os
from tqdm import tqdm
import torch
import numpy as np


def trainer(args, dataset, model):
    for e in range(1, args.max_epoch+1):
        # Training
        train_step(args=args, epoch=e, model=model, train_loader=dataset["train"])
        # Validation
        val_step(args=args, epoch=e, model=model, val_loader=dataset["valid"])
        # Save model
        if e % args.ckpt_interval == 0:
            torch.save(
                model.facial_animator.state_dict(), 
                os.path.join(args.save_model_path, 'epoch-{}_animator.pth'.format(e))
            )
            torch.save(
                model.lipreader.state_dict(), 
                os.path.join(args.save_model_path, 'epoch-{}_lipread.pth'.format(e))
            )


def train_step(args, epoch, model, train_loader):
    model.train_mode()
    model.optimizer.zero_grad()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for i, data_dict in pbar:

        # Load data
        audio = data_dict['audio'].to(device="cuda")
        vertice = data_dict['vertice'].to(device="cuda")
        template = data_dict['template'].to(device="cuda")
        one_hot = data_dict['one_hot'].to(device="cuda")
        text_token = data_dict['text_token'].to(torch.int64).to(device="cuda")
        waveform = data_dict['waveform'].transpose(0, 1).to(device="cuda")

        loss = model(audio, template, vertice, one_hot, waveform, text_token)

        # Update
        loss.backward()
        if i % args.gradient_accumulation_steps==0:
            model.optimizer.step()
            model.optimizer.zero_grad()

        # Due to memory issue
        torch.cuda.empty_cache()

        # Logging
        pbar.set_description(f"(Epoch {epoch} | Loss {loss.item():.8f}")


def val_step(args, epoch, model, val_loader):
    model.eval_mode()
    pbar = tqdm(enumerate(val_loader),total=len(val_loader))

    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    valid_loss_log = []

    for i, data_dict in pbar:

        # Load data
        audio = data_dict['audio'].to(device="cuda")
        vertice = data_dict['vertice'].to(device="cuda")
        template = data_dict['template'].to(device="cuda")
        one_hot_all = data_dict['one_hot'].to(device="cuda")
        file_name = data_dict['file_name'][0].split(".")[0]
        text_token = data_dict['text_token'].to(torch.int64).to(device="cuda")
        waveform = data_dict['waveform'].transpose(0, 1).to(device="cuda")

        train_subject = "_".join(file_name[0].split("_")[:-1])

        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:,iter,:]
            loss = model(audio, template, vertice, one_hot, waveform, text_token)
            valid_loss_log.append(loss.item())
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:,iter,:]
                loss = model(audio, template, vertice, one_hot, waveform, text_token)
                valid_loss_log.append(loss.item())
                torch.cuda.empty_cache()

        # Due to memory issue
        torch.cuda.empty_cache()

    valid_loss = np.mean(valid_loss_log)
    print(f"Epoch {epoch} | Validation Loss (avg.) {valid_loss:.8f}")