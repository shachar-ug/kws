import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.wav2vec2 = bundle.get_model()  # .to(device)
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def forward(self, x):

        # wav2vec2 inference
        # INPUT
        #   x: [batch, N_speakers, M_utterances, time] = [1, 64, 10, 0.5sec * 16khz = 8000]
        #   wavs: [N_speakers * M_utterances, time] = [640, 8000]
        # OUTPUT
        #   features: list of 12 layers [batch, N_speakers * M_utterances, time, feature size]
        #
        N_speakers, M_utterances, _ = x.shape  # _ = time vector size
        wavs = x.reshape(N_speakers * M_utterances, -1)
        # with torch.inference_mode():
        features, _ = self.wav2vec2.extract_features(wavs)
        x = features[-1]

        return x
