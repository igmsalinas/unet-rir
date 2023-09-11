import os
import random
import zipfile

import numpy as np
from tqdm import tqdm
from distributed_scripts.utils.rooms import UTSRoom
from distributed_scripts.utils.preprocess import Normalizer, FeatureExtractor, Loader, TensorPadder


class Dataset:

    def __init__(self, dir_dataset, dataset_name, extract=False, normalization=True, debugging=False,
                 room_characteristics=False, room=None, array=None):

        'Initialization'

        if room is None:
            room = ['All']
        if array is None:
            self.array = ['PlanarMicrophoneArray', 'CircularMicrophoneArray']
        else:
            self.array = array


        self.Anechoic_Room = None
        self.Hemi_Anechoic_Room = None
        self.Small_Room = None
        self.Medium_Room = None
        self.Large_Room = None
        self.Box_Room = None

        if room == ['All']:
            self.rooms = ['HemiAnechoicRoom', 'LargeMeetingRoom', 'MediumMeetingRoom', 'ShoeBoxRoom', 'SmallMeetingRoom']
        else:
            self.rooms = room

        self.normalizer = None
        self.extractor = None
        self.loader = None
        self.padder = None

        self.Spectrograms_Amp = []
        self.Spectrograms_Pha = []
        self.Embeddings = []
        self.characteristics = []

        self.index_ane = []
        self.index_hemi = []
        self.index_large = []
        self.index_medium = []
        self.index_shoe = []
        self.index_small = []

        self.index_in = []
        self.index_out = []

        self.dir_dataset = dir_dataset
        self.dataset_name = dataset_name

        'Constants'
        self.n_fft = 256
        self.win_length = 128
        self.hop_length = 64

        self.duration = 0.2  # in seconds
        self.sr = 48000
        self.mono = True

        self.input_shape = (144, 160)

        self.normalization = normalization
        self.debugging = debugging
        self.room_characteristics = room_characteristics

        self.seed = 500  # Seed for consistency at selecting training / validation and test datasets

        self.set_rooms()
        if extract:
            self.extract_files()

        self.set_preprocessers()
        self.load_data()

    def set_rooms(self):
        self.Anechoic_Room = UTSRoom(490, 722, 490, 722, 90, 90, 90, 90, 529, [245, 361], 45)
        self.Hemi_Anechoic_Room = UTSRoom(490, 722, 490, 722, 90, 90, 90, 90, 529, [245, 361], 52)
        self.Small_Room = UTSRoom(355, 410, 401, 378, 96, 90, 85, 88, 300, [175.5, 205], 497)
        self.Medium_Room = UTSRoom(736, 520, 650, 434.5, 81, 92, 98, 89, 300, [368, 217.5], 659)
        self.Large_Room = UTSRoom(994, 923, 1087, 1022, 81.4, 105, 81.3, 92.3, 300, [497, 486.25], 1281)
        self.Box_Room = UTSRoom(600, 1175, 600, 1175, 90, 90, 90, 90, 300, [300, 881.25], 667)

    def extract_files(self):
        # Extract wave files from zip folders from each zone

        dataset_path = self.dir_dataset + '/' + self.dataset_name + '/'
        room_folders = os.listdir(dataset_path)

        print("Extracting files in dataset...")

        for room_folder in tqdm(room_folders):
            print(f"Accessing {room_folder}...")
            room_path = os.path.join(dataset_path + f"/{room_folder}")
            zone_folders = os.listdir(room_path)
            for zone_folder in zone_folders:
                zone_path = os.path.join(room_path + f"/{zone_folder}")
                array_folders = os.listdir(zone_path)
                for array_folder in array_folders:
                    array_path = os.path.join(zone_path + f"/{array_folder}")
                    if array_path.endswith(".zip"):
                        file_name = os.path.abspath(array_path)
                        with zipfile.ZipFile(file_name, 'r') as zip_ref:
                            new_folder = file_name.replace(array_folder, '')
                            zip_ref.extractall(new_folder)
                        os.remove(file_name)

    def set_preprocessers(self):
        self.normalizer = Normalizer()
        self.extractor = FeatureExtractor(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
        self.loader = Loader(sample_rate=self.sr, duration=self.duration, mono=self.mono)
        self.padder = TensorPadder(desired_shape=self.input_shape)

    def load_data(self):
        dataset_path = self.dir_dataset + '/' + self.dataset_name
        room_folders = os.listdir(dataset_path)
        index = 0
        debugging_done = False
        print('Loading and preprocessing data...')
        for room_folder in room_folders:
            if self.debugging and debugging_done:
                break
            room_path = os.path.join(dataset_path + f"/{room_folder}")
            zone_folders = os.listdir(room_path)
            for zone_folder in zone_folders:
                if self.debugging and debugging_done:
                    break
                zone_path = os.path.join(room_path + f"/{zone_folder}")
                array_folders = os.listdir(zone_path)
                for array_folder in array_folders:
                    if self.debugging and debugging_done:
                        break

                    print(f'Accessing {room_folder} {zone_folder} {array_folder}')
                    array_path = os.path.join(zone_path + f"/{array_folder}")
                    rirs = os.listdir(array_path)
                    for rir_file in tqdm(rirs):
                        characteristics = rir_file.split('_')

                        if characteristics[0] in self.rooms and characteristics[2] in self.array:
                            characteristics[1] = characteristics[1].replace('Zone', '')
                            characteristics[2] = characteristics[2].replace('MicrophoneArray', '')
                            characteristics[3] = characteristics[3].replace('L', '')
                            characteristics[4] = characteristics[4].replace('M', '')
                            characteristics[4] = characteristics[4].replace('.wav', '')

                            rir_path = os.path.join(array_path + f"/{rir_file}")

                            embedding = self.get_embedding(characteristics, index)
                            amp, phase = self.preprocess(rir_path)
                            self.Spectrograms_Amp.append(amp)
                            self.Spectrograms_Pha.append(phase)

                            self.Embeddings.append(embedding)

                            if self.room_characteristics:
                                self.characteristics.append(characteristics)

                            index += 1
                            debugging_done = True
                        else:
                            debugging_done = False

        # Initial shuffle
        self.index_in = self.index_hemi + self.index_large + self.index_medium + self.index_small + self.index_shoe

        random.Random(self.seed).shuffle(self.index_hemi)
        random.Random(self.seed).shuffle(self.index_large)
        random.Random(self.seed).shuffle(self.index_medium)
        random.Random(self.seed).shuffle(self.index_small)
        random.Random(self.seed).shuffle(self.index_shoe)

        self.index_out = self.index_hemi + self.index_large + self.index_medium + self.index_small + self.index_shoe


    def get_embedding(self, characteristics, index):

        global embedding

        if characteristics[0] == "AnechoicRoom":
            embedding = self.Anechoic_Room.return_embedding(characteristics)

        elif characteristics[0] == "HemiAnechoicRoom":
            embedding = self.Hemi_Anechoic_Room.return_embedding(characteristics)
            self.index_hemi.append(index)

        elif characteristics[0] == "LargeMeetingRoom":
            embedding = self.Large_Room.return_embedding(characteristics)
            self.index_large.append(index)

        elif characteristics[0] == "MediumMeetingRoom":
            embedding = self.Medium_Room.return_embedding(characteristics)
            self.index_medium.append(index)

        elif characteristics[0] == "ShoeBoxRoom":
            embedding = self.Box_Room.return_embedding(characteristics)
            self.index_shoe.append(index)

        elif characteristics[0] == "SmallMeetingRoom":
            embedding = self.Small_Room.return_embedding(characteristics)
            self.index_small.append(index)

        return embedding

    def preprocess(self, rir_path):
        wav = self.loader.load(rir_path)
        amp, phase = self.extractor.extract(wav)
        if self.normalization:
            norm_amp, norm_phase = self.normalizer.normalize(amp, phase)
            padded_amp, padded_phase = self.padder.pad_amp_phase(norm_amp, norm_phase)
        else:
            padded_amp, padded_phase = self.padder.pad_amp_phase(amp, phase)

        return padded_amp, padded_phase

    def return_characteristics(self):
        if self.room_characteristics:
            return self.characteristics
        else:
            return None

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.Spectrograms_Amp)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        idx = index

        # Get features from dataset
        amp = self.Spectrograms_Amp[idx]
        phase = self.Spectrograms_Pha[idx]
        emb = self.Embeddings[idx]
        return amp, phase, emb


if __name__ == "__main__":
    dataset = Dataset('../../../datasets', 'room_impulse',
                      room=['All'],
                      array=['PlanarMicrophoneArray'],
                      debugging=False)
