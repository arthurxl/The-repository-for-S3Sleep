import numpy as np

from tools import covert_label


def load_npz_file(npz_file, EEG, select_label):
    """Load data and labels from a npz file."""
    with np.load(npz_file, allow_pickle=True) as f:
        # data = f[EEG][:, :, 0]
        labels = f[select_label]
        data = f[EEG]
        # labels = covert_label(labels)
        sampling_rate = 125
    data = np.squeeze(data)
    # data = data[:, np.newaxis, :]
    return data, labels, sampling_rate


def load_npz_list_files(npz_files, EEG, select_label):
    """Load data and labels from list of npz files."""
    data = []
    labels = []
    fs = None
    for npz_f in npz_files:
        print("Loading {} ...".format(npz_f))
        tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f, EEG, select_label)
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")
        # Casting
        tmp_data = tmp_data.astype(np.float32)
        tmp_labels = tmp_labels.astype(np.int32)
        data.append(tmp_data)
        labels.append(tmp_labels)
    return data, labels


def load_edf_file(edf_files, labels_npy):
    data = []
    labels = []
    for edf_f, label_edf in zip(edf_files, labels_npy):
        print("Loading {} ...".format(edf_f))
        temp_data = np.load(edf_f).squeeze(1)
        label = np.load(label_edf)
        data.append(temp_data)
        labels.append(label)
    return data, labels
