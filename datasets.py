import os
from collections import defaultdict
from pathlib import Path
from random import choices, randint, sample
from typing import Optional, Tuple, Union

import ipdb
import torch
import torchaudio
from more_itertools import locate
from torch import Tensor
from torchaudio.datasets import SPEECHCOMMANDS
from tqdm import tqdm

global WORDS
WORDS = set()


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [
                    os.path.normpath(os.path.join(self._path, line.strip()))
                    for line in fileobj
                ]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


# the dataset should retrieve part of a single word
# Accepts an input of size (N, M, D)
#     N is the number of different "speakers" in the batch (speaker - same speaker diff word)
#     M is the number of utterances per speaker
#     D is the dimensionality of the embedding vector (e.g. d-vector)
class SCEmbed(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, M_utterances: int = 3, N_speakers: int = 10):
        global WORDS
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [
                    Path(os.path.normpath(os.path.join(self._path, line.strip())))
                    for line in fileobj
                ]

        assert N_speakers > 0
        self.N_speakers = N_speakers
        self.M_utterances = M_utterances
        self.dt = 0.5  # [sec]

        if not WORDS:
            WORDS = set()
            for w in self._walker:
                WORDS.add(Path(w).parts[-2])
            WORDS = sorted(list(WORDS))

        self._num2words = dict(enumerate(WORDS))
        self._words2num = {v: k for k, v in self._num2words.items()}

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

        self._walker = [Path(w) for w in self._walker]
        words = list(
            map(
                lambda x: (x.parts[-2] + "_" + x.stem.split("_")[0], str(x)),
                self._walker,
            )
        )

        self._speakers = defaultdict(list)
        self._walker = []
        i = 0
        for candidate, f in words:
            info = torchaudio.info(f)
            length = info.num_frames / info.sample_rate
            if length < 0.5:  # [sec]
                continue

            self._walker.append(f)
            self._speakers[candidate].append(i)
            i += 1

        self._speakers = dict(self._speakers)

        # speakers id is a list of unique (speaker, work) such as zero_#
        # zero_ffbb695d -> [105567, 105568, 105569]
        # zero_ffd2ba2f -> [105570, 105571, 105572, 105573, 105574]
        # where the [] ids are indices of SPEECHCOMMANDS dataset
        self._speakers_id = list(self._speakers)

    def __len__(self) -> int:
        return len(self._speakers_id)

    def _create_samples(self, n):
        """_summary_

        Args:
            n (int): sample n out of len of speaker ids

        Returns:
            _type_: self.M utterances
        """
        remaining = self.M_utterances

        files_ids = choices(
            self._speakers[self._speakers_id[n]],
            k=min(len(self._speakers[self._speakers_id[n]]), self.M_utterances),
        )

        remaining = self.M_utterances - len(files_ids)
        files_ids += choices(self._speakers[self._speakers_id[n]], k=remaining)
        y = []
        x = []
        for file_id in files_ids:
            (
                waveform,
                sample_rate,
                label,
                speaker_id,
                utterance_number,
            ) = super().__getitem__(file_id)

            width = int(self.dt * sample_rate)
            start = randint(0, waveform.shape[1] - width)
            waveform = waveform[:, start : start + width]
            x.append(waveform)
            # y.append(self._words2num[label])
        # single label
        x = torch.cat(x)  # [M_utterances, t]
        y = self._words2num[label]
        return (x, y)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        x_pos, y_pos = self._create_samples(n)
        neg_samples_ids = list(range(0, max(0, n - 1))) + list(
            range(min(n + 1, len(self)), len(self))
        )
        x = [x_pos.unsqueeze(0)]
        y = [y_pos]
        for k in sample(neg_samples_ids, k=self.N_speakers - 1):
            xk, yk = self._create_samples(k)
            x.append(xk.unsqueeze(dim=0))
            y.append(yk)
        else:
            x = torch.cat(x)
            y = torch.tensor(y)
        return x, y


class SpeechCommandsUtterances(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, M_utterances: int = 10):
        global WORDS
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [
                    Path(os.path.normpath(os.path.join(self._path, line.strip())))
                    for line in fileobj
                ]

        self.M_utterances = M_utterances
        self.dt = 0.5  # [sec]

        if not WORDS:
            WORDS = set()
            for w in self._walker:
                WORDS.add(Path(w).parts[-2])
            WORDS = sorted(list(WORDS))

        self._num2words = dict(enumerate(WORDS))
        self._words2num = {v: k for k, v in self._num2words.items()}

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

        self._walker = [Path(w) for w in self._walker]
        words = list(
            map(
                lambda x: (x.parts[-2] + "_" + x.stem.split("_")[0], str(x)),
                self._walker,
            )
        )

        self._speakers = defaultdict(list)
        self._walker = []
        i = 0
        for candidate, f in words:
            info = torchaudio.info(f)
            length = info.num_frames / info.sample_rate
            if length < 0.5:  # [sec]
                continue

            self._walker.append(f)
            self._speakers[candidate].append(i)
            i += 1

        self._speakers = dict(self._speakers)

        # speakers id is a list of unique (speaker, work) such as zero_#
        # zero_ffbb695d -> [105567, 105568, 105569]
        # zero_ffd2ba2f -> [105570, 105571, 105572, 105573, 105574]
        # where the [] ids are indices of SPEECHCOMMANDS dataset
        self._speakers_id = list(self._speakers)

    def __len__(self) -> int:
        return len(self._speakers_id)

    def __getitem__(self, n):
        """_summary_

        Args:
            n (int): sample n out of len of speaker ids

        Returns:
            _type_: self.M utterances
        """
        files_ids = choices(
            self._speakers[self._speakers_id[n]],
            k=min(len(self._speakers[self._speakers_id[n]]), self.M_utterances),
        )

        remaining_uttr = self.M_utterances - len(files_ids)
        files_ids += choices(self._speakers[self._speakers_id[n]], k=remaining_uttr)
        x, y = [], []
        for file_id in files_ids:
            (
                waveform,
                sample_rate,
                label,
                speaker_id,
                utterance_number,
            ) = super().__getitem__(file_id)

            width = int(self.dt * sample_rate)
            start = randint(0, waveform.shape[1] - width)
            waveform = waveform[:, start : start + width]
            x.append(waveform)

        # single label
        x = torch.cat(x)  # [M_utterances, t]
        y = self._words2num[label]
        return (x, y)

    # def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
    #     x_pos, y_pos = self._create_samples(n)
    #     neg_samples_ids = list(range(0, max(0, n - 1))) + list(
    #         range(min(n + 1, len(self)), len(self))
    #     )
    #     x = [x_pos.unsqueeze(0)]
    #     y = [y_pos]
    #     for k in sample(neg_samples_ids, k=self.N_speakers - 1):
    #         xk, yk = self._create_samples(k)
    #         x.append(xk.unsqueeze(dim=0))
    #         y.append(yk)
    #     else:
    #         x = torch.cat(x)
    #         y = torch.tensor(y)
    #     return x, y


def sce_collate(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """

    x, y = zip(*data)
    x = [torch.cat(_x) for _x in x]
    y = [torch.tensor(_y) for _y in y]
    return x, y
