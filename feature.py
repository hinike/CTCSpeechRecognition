from features import logfbank
import scipy.io.wavfile as wav
from optparse import OptionParser
import os
import numpy as np
import lmdb
import shutil
import sys

parser = OptionParser()
parser.add_option("--path1", dest="PATH1", type="string")
parser.add_option("--path2", dest="PATH2", type="string")
parser.add_option("--path3", dest="PATH3", type="string")
(options, args) = parser.parse_args()


def wav2logfbank(root_path, index_path, out_dir):
    """
        generate logfbank (nfilt X frames) of wav file
    """

    assert os.path.exists(out_dir), 'out_dir does not exist'
    out_dir = out_dir + '/' + 'spect/'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir) # remove the old lmdb
        os.mkdir(out_dir)

    cnts = 1 #index starts with 1
    env = lmdb.open(out_dir, map_size=10**12) # 1TB map size; no penalty
    for line in open(index_path):
        _path, _, __ = line.split('@')
        wave_path = root_path + '/' + _path
        assert '.wav' in wave_path, 'make sure wav files end with suffix .wav'

        print 'processing: ', wave_path, cnts
        (rate, sig) = wav.read(wave_path)
        feat = logfbank(sig, samplerate=rate).T

        with env.begin(write=True) as txn:
            txn.put(str(cnts), np.float32(feat.clip(0)).tostring())

        cnts += 1

    print cnts, 'items are saved'

if __name__ == '__main__':
    wav2logfbank(options.PATH1, options.PATH2, options.PATH3)








