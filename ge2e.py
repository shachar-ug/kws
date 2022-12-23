import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from utils import calc_loss, get_centroids, get_cossim
import ipdb


def collate_batch(batch):
    ipdb.set_trace()


class BatchSpeechEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

        # self.device = device
        self.LSTM_stack = nn.LSTM(768, 768, num_layers=3, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(768, 256)

        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.wav2vec2 = bundle.get_model()  # .to(device)
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def forward(self, x):

        # wav2vec2 inference
        # INPUT
        #   x: [N_speakers, M_utterances, time] = [1, 64, 10, 0.5sec * 16khz = 8000]
        #   wavs: [N_speakers * M_utterances, time] = [640, 8000]
        # OUTPUT
        #   features: list of 12 layers [batch, N_speakers * M_utterances, time, feature size]
        #
        N_speakers, M_utterances, _ = x.shape  # _ = time vector size
        wavs = x.reshape(N_speakers * M_utterances, -1)
        # with torch.inference_mode():
        features, _ = self.wav2vec2.extract_features(wavs)
        x = features[-1]

        x, _ = self.LSTM_stack(x)  # (batch, frames, :)
        # only use last frame
        x = x[:, x.size(1) - 1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)

        x = x.reshape(N_speakers, M_utterances, -1)
        return x


class SpeechEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

        # self.device = device
        self.LSTM_stack = nn.LSTM(768, 768, num_layers=3, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(768, 256)

        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.wav2vec2 = bundle.get_model()  # .to(device)
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def forward(self, x):

        # wav2vec2 inference
        # INPUT
        #   x: [N_speakers, M_utterances, time] = [1, 64, 10, 0.5sec * 16khz = 8000]
        #   wavs: [N_speakers * M_utterances, time] = [640, 8000]
        # OUTPUT
        #   features: list of 12 layers [N_speakers * M_utterances, time, feature size]
        #
        N_speakers, M_utterances, _ = x.shape  # _ = time vector size
        wavs = x.reshape(N_speakers * M_utterances, -1)
        # with torch.inference_mode():
        features, _ = self.wav2vec2.extract_features(wavs)
        x = features[-1]

        x, _ = self.LSTM_stack(x)  # (batch, frames, n_mels)
        # only use last frame
        x = x[:, x.size(1) - 1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)

        x = x.reshape(N_speakers, M_utterances, -1)
        return x


class SpeechEmbedderLit(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.LSTM_stack = nn.LSTM(768, 768, num_layers=3, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(768, 256)

        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.wav2vec2 = bundle.get_model()  # .to(device)
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        self.ge2e_loss = GE2ELoss()

    def forward(self, x):

        # wav2vec2 inference
        # INPUT
        #   x: [N_speakers, M_utterances, time] = [1, 64, 10, 0.5sec * 16khz = 8000]
        #   wavs: [N_speakers * M_utterances, time] = [640, 8000]
        # OUTPUT
        #   features: list of 12 layers [N_speakers * M_utterances, time, feature size]
        #
        N_speakers, M_utterances, _ = x.shape  # _ = time vector size
        wavs = x.reshape(N_speakers * M_utterances, -1)
        # with torch.inference_mode():
        features, _ = self.wav2vec2.extract_features(wavs)
        x = features[-1]

        x, _ = self.LSTM_stack(x)  # (batch, frames, n_mels)
        # only use last frame
        x = x[:, x.size(1) - 1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)

        x = x.reshape(N_speakers, M_utterances, -1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        embedding = self(x)
        loss = self.ge2e_loss(embedding)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        embedding = self(x)
        loss = self.ge2e_loss(embedding)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        # https://github.com/Lightning-AI/lightning/issues/7576
        # optimizer = torch.optim.SGD(
        #     [{"params": self.parameters()}, {"params": self.ge2e_loss.parameters()}],
        #     lr=0.01,
        # )
        return torch.optim.Adam(self.parameters(), lr=0.01)


# class SpeechEmbedderLT(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.LSTM_stack = nn.LSTM(768, 768, num_layers=3, batch_first=True)
#         for name, param in self.LSTM_stack.named_parameters():
#             if "bias" in name:
#                 nn.init.constant_(param, 0.0)
#             elif "weight" in name:
#                 nn.init.xavier_normal_(param)
#         self.projection = nn.Linear(768, 256)

#     def forward(self, x):
#         x, _ = self.LSTM_stack(x.float())  # (batch, frames, n_mels)
#         # only use last frame
#         x = x[:, x.size(1) - 1]
#         x = self.projection(x.float())
#         x = x / torch.norm(x, dim=1).unsqueeze(1)
#         return x


class GE2ELoss(nn.Module):
    def __init__(self):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0), requires_grad=True)

    def forward(self, embeddings):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w * cossim + self.b
        loss, _ = calc_loss(sim_matrix)
        return loss


class GE2ELossV0(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0, loss_method="softmax"):
        """
        Implementation of the Generalized End-to-End loss defined in https://arxiv.org/abs/1710.10467 [1]

        Accepts an input of size (N, M, D)

            where N is the number of speakers in the batch,
            M is the number of utterances per speaker,
            and D is the dimensionality of the embedding vector (e.g. d-vector)

        Args:
            - init_w (float): defines the initial value of w in Equation (5) of [1]
            - init_b (float): definies the initial value of b in Equation (5) of [1]
        """
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.loss_method = loss_method

        assert self.loss_method in ["softmax", "contrast"]

        if self.loss_method == "softmax":
            self.embed_loss = self.embed_loss_softmax
        if self.loss_method == "contrast":
            self.embed_loss = self.embed_loss_contrast

    def calc_new_centroids(self, dvecs, centroids, spkr, utt):
        """
        Calculates the new centroids excluding the reference utterance
        """
        excl = torch.cat((dvecs[spkr, :utt], dvecs[spkr, utt + 1 :]))
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, centroids):
        """
        Make the cosine similarity matrix with dims (N,M,N)
        """
        cos_sim_matrix = []
        for spkr_idx, speaker in enumerate(dvecs):
            cs_row = []
            for utt_idx, utterance in enumerate(speaker):
                new_centroids = self.calc_new_centroids(
                    dvecs, centroids, spkr_idx, utt_idx
                )
                # vector based cosine similarity for speed
                cs_row.append(
                    torch.clamp(
                        torch.mm(
                            utterance.unsqueeze(1).transpose(0, 1),
                            new_centroids.transpose(0, 1),
                        )
                        / (torch.norm(utterance) * torch.norm(new_centroids, dim=1)),
                        1e-6,
                    )
                )
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        """
        Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
        """
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                L_row.append(-F.log_softmax(cos_sim_matrix[j, i], 0)[j])
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def embed_loss_contrast(self, dvecs, cos_sim_matrix):
        """ 
        Calculates the loss on each embedding $L(e_{ji})$ by contrast loss with closest centroid
        """
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j, i])
                excl_centroids_sigmoids = torch.cat(
                    (centroids_sigmoids[:j], centroids_sigmoids[j + 1 :])
                )
                L_row.append(
                    1.0
                    - torch.sigmoid(cos_sim_matrix[j, i, j])
                    + torch.max(excl_centroids_sigmoids)
                )
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def forward(self, dvecs):
        """
        Calculates the GE2E loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        """
        # Calculate centroids
        centroids = torch.mean(dvecs, 1)

        # Calculate the cosine similarity matrix
        cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        L = self.embed_loss(dvecs, cos_sim_matrix)
        return L.sum()

