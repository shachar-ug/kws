import time

import ipdb
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets
from ge2e import (
    GE2ELoss,
    SpeechEmbedder,
    BatchSpeechEmbedder,
    collate_batch,
    SpeechEmbedderLit,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def trainer_pl(device=device):
    N = 64  # Number of speakers in a batch
    M = 10  # Number of utterances for each speaker
    D = 256  # Dimensions of the speaker embeddings, such as a d-vector or x-vector

    # Create training and testing split of the data. We do not use validation in this tutorial.
    # train_dataset = datasets.SCEmbed("training", M_utterances=M, N_speakers=N)
    # train_loader = DataLoader(train_dataset, batch_size=N)

    # https://teddykoker.com/2020/12/dataloader/
    train_dataset = datasets.SpeechCommandsUtterances("training", M_utterances=M)
    val_dataset = datasets.SpeechCommandsUtterances("validation", M_utterances=M)

    train_loader = DataLoader(train_dataset, batch_size=N, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=N, shuffle=False, num_workers=16)

    trainer = pl.Trainer(max_epochs=10, accelerator=device)
    model = SpeechEmbedderLit()

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def trainer(device=device):
    N = 64
    # Number of speakers in a batch
    M = 10  # Number of utterances for each speaker
    D = 256  # Dimensions of the speaker embeddings, such as a d-vector or x-vector

    # Create training and testing split of the data. We do not use validation in this tutorial.
    # train_dataset = datasets.SCEmbed("training", M_utterances=M, N_speakers=N)
    # train_loader = DataLoader(train_dataset, batch_size=N)

    # https://teddykoker.com/2020/12/dataloader/
    train_dataset = datasets.SpeechCommandsUtterances("training", M_utterances=M)
    train_dataset = datasets.SpeechCommandsUtterances("validation", M_utterances=M)

    train_loader = DataLoader(train_dataset, batch_size=N, shuffle=True, num_workers=16)

    net = BatchSpeechEmbedder().to(device)
    ge2e_loss = GE2ELoss().to(device)

    optimizer = torch.optim.SGD(
        [{"params": net.parameters()}, {"params": ge2e_loss.parameters()}], lr=0.01
    )

    x, y = next(iter(train_loader))
    x = x[0]
    x = x.to(device)
    y = y.to(device)

    # out = se(x)
    # v = loss(out)
    # https://github.com/funcwj/ge2e-speaker-verification/blob/master/ge2e/trainer.py
    net.train()
    EPOCHS = 5000
    for n_epoch in range(EPOCHS):
        total_loss = 0

        for n_batch, (x, y) in enumerate(train_loader):
            torch.cuda.empty_cache()
            # x = x[0].to(device)
            x = x.to(device)
            x_est = net(x)
            loss = ge2e_loss(x_est)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.detach().cpu().numpy()

            if (n_batch + 1) % 10 == 0:
                s = ""
                s += f"{time.ctime()}\t"
                s += f"iteration={n_batch}/{len(train_dataset)} "
                s += f"loss={loss} "
                s += f"mean loss = {total_loss/(n_batch+1)}"
                print(s)

            del loss
