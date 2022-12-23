import ipdb
import torch
import torchaudio
from torch.utils.data import DataLoader

import datasets
from ge2e import GE2ELoss, SpeechEmbedder
from train import trainer, trainer_pl

device = "cuda" if torch.cuda.is_available() else "cpu"


if True:
    trainer_pl(device)
else:
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    for param in model.parameters():
        param.requires_grad = False
    # x.shape: [batch, N_speakers, M_utterances, time]
    N = 64  # Number of speakers in a batch
    M = 10  # Number of utterances for each speaker
    D = 256  # Dimensions of the speaker embeddings, such as a d-vector or x-vector

    # Create training and testing split of the data. We do not use validation in this tutorial.
    train_set = datasets.SCEmbed("training", M_utterances=M, N_speakers=N)
    se = SpeechEmbedder(device=device)
    train_loader = DataLoader(
        train_set, batch_size=1
    )  # , collate_fn=datasets.sce_collate)

    x, y = next(iter(train_loader))
    x = x.to(device)

    # criterion = GE2ELoss(
    #     init_w=10.0, init_b=-5.0, loss_method="contrast"
    # )  # for contrast loss

    wavs = x.reshape(x.shape[0] * x.shape[1] * x.shape[2], -1)
    x1 = wavs.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

    with torch.inference_mode():
        features, _ = model.extract_features(wavs)

    # output shape
    # x: [batch, N_speakers, M_utterances, time] = [1, 64, 10, 0.5sec * 16khz = 8000]
    # wavs: [batch * N_speakers * M_utterances, time] = [640, 8000]
    # features: list of 12 layers [batch * N_speakers * M_utterances, time, feature size]
    #                             (batch, time frame, feature dimension)
    # https://pytorch.org/audio/main/_modules/torchaudio/models/wav2vec2/model.html
    # features[-1] is the last layer (12th)
    ipdb.set_trace()

    out = se(features[-1])
    out = out.reshape(N, M, -1)
    ipdb.set_trace()

    # wants (Speaker, Utterances, embedding)
    # (N, M, embedding_size)
