import librosa
import pandas as pd
import soundfile as sf
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import librosa
import numpy as np
from scipy.io import wavfile
from IPython.display import Audio
import time
from gammatone import gtgram
from scipy.fft import dct

meta = pd.read_csv("meta/esc50.csv")
meta


def log_function_call(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()

        # Log function name
        print(f"Function Name: {func.__name__}")
        # Call the original function
        result = func(*args, **kwargs)
        # Log time taken
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time Taken: {elapsed_time:.4f} seconds")

        return result

    return wrapper


@log_function_call
def train_test_split_util():
    train, test = None, None
    if not set(["train_meta.csv", "test_meta.csv"]) - set(os.listdir("meta")) == set():
        train, test = train_test_split(
            meta, test_size=0.2, stratify=meta["target"], random_state=42
        )
        train.to_csv("meta/train_meta.csv", index=False)
        test.to_csv("meta/test_meta.csv", index=False)
    else:
        train, test = pd.read_csv("meta/train_meta.csv"), pd.read_csv(
            "meta/test_meta.csv"
        )
    return (
        train["filename"].to_list(),
        test["filename"].to_list(),
        train["target"].to_list(),
        test["target"].to_list(),
    )

def gtcc(y, sr):
    
    window_size = 0.025  # 25 ms window size
    hop_size = 0.01  # 10 ms hop size
    num_mel_filters = 40
    
    gtm = gtgram.gtgram(
        wave=y,
        fs=sr,
        window_time=window_size,
        hop_time=hop_size,
        channels=num_mel_filters,
        f_min=50)
    # print(gtm.shape)
    gtm_log = np.log(gtm+1e-10)
    
    final_gtcc = dct(gtm_log, axis=0)
    
    # print(final_gtcc.shape)
    
    return final_gtcc[1:13,]

@log_function_call
def extract_features(y, sr_list=None, ft="mfcc", sr=22050, n_mfcc=13):
    sr = sr
    if type(y[0]) == np.ndarray:
        return_data = []
        if ft == "mfcc":
            for wave in y:
                mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=n_mfcc)
                return_data.append(np.array(mfcc))
        elif ft == "mel":
            for wave in y:
                # print(sr)
                mel_spectrogram = librosa.feature.melspectrogram(y=wave, sr=sr, hop_length=345, n_mels=n_mfcc, fmin=50, fmax=14000)
                log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
                return_data.append(np.array(log_mel_spectrogram))
        elif ft == "chroma":
            for wave in y:
                chroma = librosa.feature.chroma_stft(y=wave, sr=sr)
                return_data.append(np.array(chroma))
        elif ft == "spec":
            for wave in y:
                spect_con = librosa.feature.spectral_contrast(y=wave, sr=sr)
                return_data.append(spect_con)
        elif ft == "mfcc_delta":
            for wave in y:
                mfcc_delta = librosa.feature.delta(
                    librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=n_mfcc)
                )
                return_data.append(mfcc_delta)
        elif ft == "chroma_cqt":
            for wave in y:
                chroma_cqt = librosa.feature.chroma_cqt(y=wave, sr=sr)
                return_data.append(chroma_cqt)
        elif ft == "mfcc_delta2":
            for wave in y:
                mfcc_delta = librosa.feature.delta(
                    librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=n_mfcc), order=2
                )
                return_data.append(mfcc_delta)
        elif ft == "spectrogram":
            for wave in y:
                spec_data = librosa.stft(wave)
                return_data.append(spec_data)
        elif ft == "log_spectrogram":
            print("log_spectrogram")
            print(sr,n_mfcc)
            for wave in y:
                mel_spectrogram = librosa.feature.melspectrogram(y=wave, sr=sr, n_mels=n_mfcc)
                log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
                return_data.append(log_mel_spectrogram)
        elif ft == "gtcc":
            print("Gammatone Cepstral Coefficients")
            for wave in y:
                print(sr)
                gtcc_data = gtcc(y=wave, sr=sr)
                return_data.append(gtcc_data)
        return np.array(return_data)
    else:
        if ft == "mfcc":
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[2:14, ]
            return np.array(mfccs)
        elif ft == "chroma":

            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            return np.array(chroma)
        elif ft == "spec":
            spect_con = librosa.feature.spectral_contrast(y=y, sr=sr)
            return spect_con
        elif ft == "mfcc_delta":
            mfcc_delta = librosa.feature.delta(
                librosa.feature.mfcc(y=y, sr=sr, n_mfcc=7)
            )
            return np.array(mfcc_delta)
        elif ft == "chroma_cqt":
            chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
            return np.array(chroma_cqt)
        elif ft == "mfcc_delta2":
            mfcc_delta = librosa.feature.delta(
                librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), order=2
            )
            return np.array(mfcc_delta)
        elif ft == "gtcc":
            gtcc_data = gtcc(y=y, sr=sr)
            return np.array(gtcc_data)
        elif ft == "mel":
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=345, n_mels=n_mfcc, fmin=50, fmax=14000)
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            return np.array(log_mel_spectrogram)


def load_and_preprocess_data(data_list):
    print("load preprocessing start")
    feature_mfcc = [extract_features(data, ft="mfcc") for data in data_list]
    # features_chroma = [extract_features(file_path, type="chroma") for file_path in file_paths]
    features_spec = [extract_features(data, ft="spec") for data in data_list]
    features_mfcc_delta = [
        extract_features(data, ft="mfcc_delta") for data in data_list
    ]
    # features_chroma_cqt = [extract_features(file_path, type="chroma_cqt") for file_path in file_paths]
    features_mfcc_delta2 = [
        extract_features(data, ft="mfcc_delta2") for data in data_list
    ]

    # Convert lists to numpy arrays
    feature_mfcc = np.array(feature_mfcc)
    features_spec = np.array(features_spec)
    features_mfcc_delta = np.array(features_mfcc_delta)
    # features_chroma_cqt = np.array(features_chroma_cqt)
    features_mfcc_delta2 = np.array(features_mfcc_delta2)
    print("load preprocessing complete")
    return (
        feature_mfcc,
        features_spec,
        features_mfcc_delta,
    )


@log_function_call
def get_waves(file_names, base_path="audio/"):
    audio_wave_list, sr_list = [], []
    for file_name in file_names:
        audio_path = base_path + str(file_name)
        audio, sr = librosa.load(audio_path, sr=None)
        audio_wave_list.append(audio)
        sr_list.append(sr)
    return audio_wave_list, sr_list


@log_function_call
def get_waves_5_sec(file_names, base_path="audio/"):
    audio_wave_list, sr_list = [], []
    for file_name in file_names:
        audio_path = base_path + str(file_name)
        audio, sr = librosa.load(audio_path, sr=None)
        five_second_clip = np.tile(audio, 5)

        audio_wave_list.append(five_second_clip)
        sr_list.append(sr)
    return audio_wave_list, sr_list

def augment_noise(file):
    audio_path = str(file)
    audio, sr = librosa.load(audio_path)
    # print(sr)
    # Audio(data=audio, rate=sr)

    #     # Compute mel spectrogram
    #     mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=512, hop_length=256, n_mels=70)
    #     mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    #     # Invert mel spectrogram back to time-domain signal
    #     augmented_audio = librosa.griffinlim(librosa.db_to_power(mel_spectrogram_db, ref=1), hop_length=256)
    noise = np.random.normal(0, 0.05, audio.shape)
    augmented_audio = audio + noise
    return audio, augmented_audio


# Create noisy data
def create_noisy_data(files, target):
    print("Start Noise creation")
    noisy_data, noisy_targets = [], []
    train_data = []
    for i in range(len(files)):
        audio, augment_audio = augment_noise(files[i])
        noisy_data.append(augment_audio)
        train_data.append(audio)
        noisy_targets.append(target[i])
    print("Complete Noise creation")
    return train_data, noisy_data, noisy_targets


def create_test_data(files):
    data = []
    for i in files:
        audio_path = str(i)
        audio, sr = librosa.load(audio_path)
        data.append(audio)
    return data


def split_audio(files, targets, split_time=1, prefix_metadata_file="train"):
    if (
        set([f"{prefix_metadata_file}_split_{split_time}.csv"])
        - set(os.listdir("meta"))
        == set()
    ):
        data = pd.read_csv(f"meta/{prefix_metadata_file}_split_{split_time}.csv")
        return data["filename"].to_list(), data["target"].to_list()
    new_files, new_target = [], []
    for i in range(len(files)):
        audio_file = "audio" + "/" + files[i]
        y, sr = librosa.load(audio_file, sr=None)
        # Set the duration of each segment (in seconds)
        segment_duration = 1
        # Calculate the total number of segments
        total_segments = int(np.ceil(len(y) / (sr * segment_duration)))
        new_data = []
        # Iterate through segments
        for j in range(total_segments):
            # Calculate start and end indices for the current segment
            start_index = int(j * sr * segment_duration)
            end_index = int((j + 1) * sr * segment_duration)
            # Extract the segment from the audio
            segment = y[start_index:end_index]

            contains_non_zero = np.any(segment != 0)
            if not contains_non_zero:
                continue
            file_name = (
                files[i].split(".")[0] + "_" + "split_segment_" + str(j + 1) + ".wav"
            )

            sf.write("audio/" + str(file_name), segment, sr, subtype="PCM_24")
            new_files.append(file_name)
            new_target.append(targets[i])
    dataframe = pd.DataFrame({"filename": new_files, "target": new_target})
    dataframe.to_csv(f"meta/{prefix_metadata_file}_split_{split_time}.csv", index=False)
    return new_files, new_target

def split_audio2(files, targets, split_time=1, prefix_metadata_file="train"):
    if (
        set([f"{prefix_metadata_file}_split_{split_time}.csv"])
        - set(os.listdir("meta"))
        == set()
    ):
        data = pd.read_csv(f"meta/{prefix_metadata_file}_split_{split_time}.csv")
        return data["filename"].to_list(), data["target"].to_list()
    new_files, new_target = [], []
    for i in range(len(files)):
        audio_file = "audio" + "/" + files[i]
        y, sr = librosa.load(audio_file, sr=None)
        # Set the duration of each segment (in seconds)
        segment_duration = 1
        # Calculate the total number of segments
        total_segments = int(np.ceil(len(y) / (sr * segment_duration)))
        new_data = []
        # Iterate through segments
        for j in range(total_segments):
            # Calculate start and end indices for the current segment
            start_index = int(j * sr * segment_duration)
            end_index = int((j + 1) * sr * segment_duration)
            # Extract the segment from the audio
            segment = y[start_index:end_index]

            contains_non_zero = np.any(segment != 0)
            if not contains_non_zero:
                continue
            file_name = (
                files[i].split(".")[0] + "_" + "split_segment_" + str(j + 1) + ".wav"
            )

            sf.write("audio2/" + str(file_name), segment, sr, subtype="PCM_24")
            new_files.append(file_name)
            new_target.append(targets[i])
    dataframe = pd.DataFrame({"filename": new_files, "target": new_target})
    dataframe.to_csv(f"meta/{prefix_metadata_file}_split_{split_time}.csv", index=False)
    return new_files, new_target

def bell():
    import chime

    chime.success()



from tensorflow.keras.models import load_model
import numpy as np
def test_acc(trained_model, X_test):
    predicted = trained_model.predict(
        X_test
    )
    predicted = np.argmax(predicted, axis=1)
    testing = pd.read_csv("meta/test_split_1.csv")
    testing["filename"]
    split_parts = testing["filename"].str.split('_split_segment', expand=True)
    testing["original_filename"] = split_parts[0]
    testing["predict"] = predicted
    agg_df = testing.groupby('original_filename').agg({'target': list, 'predict': list}).reset_index()
    max_values = agg_df[['target', 'predict']].applymap(max)

    max_values.columns = ['max_target', 'max_predict']
    count_equal_max = (max_values['max_target'] == max_values['max_predict']).sum()
    acc = count_equal_max*100/len(max_values)
    return acc
