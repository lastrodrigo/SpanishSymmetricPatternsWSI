import os
from bin.bilm import dump_token_embeddings


paths = ['analogies', 'similarity']
words = set(['<s>', '</s>'])
for path in paths:
    files = os.listdir('../evaluation/' + path)

    for file_name in files:
        with open('../evaluation/' + os.path.join(path,file_name)) as f:
            lines = f.readlines()
            for line in lines:
                if path == 'similarity':
                    words.update([x.replace('\n', '') for x in line.split(',')[0:2] if x != ''])
                if path == 'analogies':
                    words.update([x.replace('\n', '') for x in line.split(' ')[0:4] if x != ''])
vocab_file = '../evaluation/vocabulary'
with open(vocab_file, 'w+') as f:
    for word in words:
        f.writelines(f'{word}\n')

# Location of pretrained LM.  Here we use the test fixtures.
datadir = '../SymPatternWSI-master/resources'
options_file = os.path.join(datadir, 'options262.json')
weight_file = os.path.join(datadir, 'weights.hdf5')

# Dump the token embeddings to a file. Run this once for your dataset.
token_embedding_file = '../evaluation/elmo_token_embeddings.hdf5'
dump_token_embeddings(
    vocab_file, options_file, weight_file, token_embedding_file
)
                    