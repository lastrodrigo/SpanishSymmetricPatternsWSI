import h5py
from scipy import spatial
from scipy.stats import spearmanr
import os

filename = 'elmo_token_embeddings.hdf5'
f = h5py.File(filename, 'r')
key = list(f.keys())[0]
with open('vocabulary', 'r') as voc:
    tokens = [x.replace('\n', '') for x in voc.readlines()]

folder_path = 'similarity'
files = os.listdir(folder_path)
for sim_file in files:
    with open(os.path.join(folder_path, sim_file)) as data:
        tests = data.readlines()

    test_scores = []
    distances = []
    for test in tests:
        split_test = test.split(',')
        if split_test[0] != '' and split_test[1] != '':
            embedding1 = f[key][tokens.index(split_test[0])]
            embedding2 = f[key][tokens.index(split_test[1])]
            test_scores.append(split_test[2])
            cosine = spatial.distance.cosine(embedding1, embedding2)
            distances.append(cosine)
    print('='*20)
    print(sim_file)
    print(spearmanr(test_scores, distances))
