"""
Ignacio Martín 2022

Code corresponding to the Bachelor Thesis "Synthesis of Room Impulse Responses by Means of Deep Learning"

Authors:
    Ignacio Martín
    José Antonio Belloch
    Gema Piñero

University Carlos III de Madrid
"""

import os
import pathlib

import librosa
import tensorflow as tf
import numpy as np
from scipy.io.wavfile import write

from distributed_scripts.utils.preprocess import Normalizer, TensorPadder


class PostProcess:
    """
    PostProcess takes a feature (Log Spectrogram and Phase) normalized and padded,
    deletes the padding and denormalizes it, computes the inverse STFT and converts it into a wav
    Steps:
    1- Loads feature and vector
    2- Deletes padding
    3- Denormalizes them
    4- Log Spec -> Spec
    5- ISTFT
    6- Converts to wav
    """

    def __init__(self, folder, algorithm=None):

        self.stft = None
        self.phase = None
        self.waveform = None

        self.normalizer = Normalizer()
        self.padder = TensorPadder((144, 160))

        if algorithm is 'gl':
            self.algorithm = 'gl'
        else:
            self.algorithm = 'ph'

        self.wav_path = "../generated_rir_distributed/" + folder + f'_{self.algorithm}'

    def post_process(self, feature, vector, des_shape=(129, 151),
                     n_fft=256, win_length=128, hop_length=64, sr=48000):
        """
        Takes a feature, its corresponding vector, the min_max normalization values, the previous shape of the STFT
        and proceeds to transform, denormalize, perform the ISTFT with the given parameters and saves the wav and STFT.

        :param feature: np.ndarray containing log magnitude and phase
        :param vector: vector list of the feature
        :param min_max: min max values used for normalization
        :param des_shape: previous shape before padding
        :param hop_length: hop length stft
        :param win_length: window length stft
        :param n_fft: number of ffts stft
        :param sr: sample rate
        """
        stft, phase = self.get_stft_phase(feature)
        stft_d, phase_d = self.de_shape(stft, phase, des_shape)
        denorm_f, denorm_p = self.denormalize(stft_d, phase_d)
        self.istft(denorm_f, denorm_p, n_fft, win_length, hop_length)
        self.save_wav(sr, vector)
        self.save_stft(feature)

        return self.waveform

    @staticmethod
    def get_stft_phase(feature):
        """
        Unstacks the feature.

        :param feature: input feature
        :return: log stft, phase
        """
        # stft, phase = np.moveaxis(feature, 2, 0)
        stft = feature[:, :, 0]
        phase = feature[:, :, 1]
        return stft, phase

    def de_shape(self, stft, phase, des_shape):
        """
        Removes the padding.

        :param stft: log magnitude stft
        :param phase: phase stft
        :param des_shape: previous shape
        :return: transformed log_stft, phase
        """
        stft_d, phase_d = self.padder.un_pad(stft, phase, des_shape)

        return stft_d, phase_d

    def denormalize(self, stft, phase):
        """
        Denormalizes the log magnitude and phase given the previous min_max.

        :param stft: input log magnitude
        :param phase: input phase
        :param min_max: min max used in normalization
        :return: denormalized feature
        """
        denorm_stft, denorm_phase = self.normalizer.denormalize(stft, phase)
        return denorm_stft, denorm_phase

    def istft(self, denorm_f, denorm_p, n_fft, win_length, hop_length):
        """
        Performs the inverse STFT given the parameters and the denormalized feature.

        :param denorm_f: denormalized magnitude
        :param denorm_p: denormalized phase
        :param hop_length: hop length
        :param win_length: window length
        :param n_fft: number of ffts
        """

        if self.algorithm == 'ph':
            conv_stft = denorm_f * (np.cos(denorm_p) + 1j * np.sin(denorm_p))
            waveform = librosa.istft(conv_stft, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        elif self.algorithm == 'gl':
            waveform = librosa.griffinlim(denorm_f, n_fft=n_fft, win_length=win_length, hop_length=hop_length)

        self.waveform = waveform

    def save_wav(self, sr, vector):
        """
        Writes wav given a sample rate and the vector corresponding to the waveform.

        :param sr: sample rate
        :param vector: information vector
        """
        vector_name = ""
        for value in vector:
            value_str = str(value)
            vector_name += "-" + value_str
        self.wav_name = "RIR" + vector_name
        self._create_directory_if_none(self.wav_path + "/rir/")
        file_path = os.path.join(self.wav_path + "/rir/", self.wav_name + ".wav")
        write(file_path, sr, self.waveform)

    def save_stft(self, feature):
        """
        Saves the log magnitude and phase generated by the model.

        :param feature: generated stft
        """
        self._create_directory_if_none(self.wav_path + "/stft/")
        file_path = os.path.join(self.wav_path + "/stft/", self.wav_name)
        np.save(file_path + ".npy", feature)

    @staticmethod
    def _create_directory_if_none(dir_path):
        """
        Creates a directory.

        :param dir_path: Path to make directory.
        """
        dir_path = dir_path
        directory = pathlib.Path(dir_path)
        if not directory.exists():
            os.makedirs(dir_path)
