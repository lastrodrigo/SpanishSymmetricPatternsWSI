#!/usr/bin/python
# -*- coding: latin-1 -*-
import logging
from elmoformanylangs import Embedder
from .bilm_interface import Bilm
import scipy
import numpy as np
import io
from tqdm import tqdm
import os
import multiprocessing
from typing import List, Dict, Tuple
from sklearn.decomposition import PCA

class ElmoManyLangs(Bilm) :

    def _load_vectors(self,fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.asarray(list(map(float, tokens[1:])))

        return data

    def __init__(self, cuda_device, model_path, vocab_path, semfit_path, batch_size=64,
                 cutoff_elmo_vocab=50000):
        super().__init__()
        logging.info(
            'creating elmo in device %d. model path %s, vocab_path %s '
            ' batch_size: %d' % (
                cuda_device, model_path, vocab_path,
                batch_size))
        self.elmo = Embedder(model_path,batch_size)

        self.batch_size = batch_size

        logging.info('warming up elmo')
        self._warm_up_elmo()

        self.elmo_word_vocab = []
        self.elmo_word_vocab_lemmatized = []

        # we prevent the prediction of these by removing their weights and their vocabulary altogether
        stop_words = {'<UNK>', '<S>', '</S>', '--', '..', '...', '....'}

        logging.info('reading elmo vocabulary')

        lines_to_remove = set()
        with open(vocab_path, encoding="utf-8") as fin:
            for idx, line in enumerate(fin):
                if idx == cutoff_elmo_vocab:
                    break
                word = line.strip().split()[0]
                if len(word) == 1 or word in stop_words:
                    lines_to_remove.add(idx)
                self.elmo_word_vocab.append(word)

        with open(vocab_path + '.lemmatized', encoding="utf-8") as fin:
            for idx, line in enumerate(fin):
                if idx == cutoff_elmo_vocab:
                    break
                word = line.strip().split()[0]
                if len(word) == 1 or word in stop_words:
                    lines_to_remove.add(idx)
                self.elmo_word_vocab_lemmatized.append(word)

        # remove stopwords
        self.elmo_word_vocab = [x for i, x in enumerate(self.elmo_word_vocab) if i not in lines_to_remove]
        self.elmo_word_vocab_lemmatized = [x for i, x in enumerate(self.elmo_word_vocab_lemmatized) if
                                           i not in lines_to_remove]

        self.semfit = self._load_vectors(semfit_path)

    def _warm_up_elmo(self):
        # running a few sentences in elmo will set it to a better state than initial zeros
        warm_up_sent = "En efecto , rematado ya su juicio , vino a dar en el más " \
                       "extraño pensamiento que jamás dio loco en el mundo ; y fue que " \
                       "le pareció convenible y necesario , así para el aumento de su honra " \
                       "como para el servicio de su república , hacerse caballero andante , e irse " \
                       "por todo el mundo con sus armas y caballo a buscar las " \
                       "aventuras y a ejercitarse en todo aquello que él había leído que " \
                       "los caballeros andantes se ejercitaban , deshaciendo todo género de agravio , y poniéndose " \
                       "en ocasiones y peligros donde , acabándolos , cobrase eterno nombre y fama .".split()
        for _ in range(3):
            _ = list(self.elmo.sents2elmo([warm_up_sent] * self.batch_size))
   
    @staticmethod
    def create_lemmatized_vocabulary_if_needed(vocab_path):
        """
        this creates a new voabulary file in the same directory as ELMo vocab where words has been lemmatized
        :param vocab_path: path to ELMo vocabulary
        :return:
        """
        if not os.path.isfile(vocab_path + '.lemmatized'):
            # if there is not lemmatized vocabulary create it
            with open(vocab_path, encoding="utf-8") as fin:
                unlem = [x.strip().split()[0] for x in fin.readlines()] #+RL
                
            logging.info('lemmatizing ELMo vocabulary')
            print('lemmatizing ELMo vocabulary')
            import spacy
            nlp = spacy.load("es", disable=['ner', 'parser']) #RL
            new_vocab = []
            for spacyed in tqdm(
                    nlp.pipe(unlem, batch_size=1000, n_threads=multiprocessing.cpu_count()),
                    total=len(unlem)):
                new_vocab.append(spacyed[0].lemma_ if spacyed[0].lemma_ != '-PRON-' else spacyed[0].lower_)
            with open(vocab_path + '.lemmatized', 'w', encoding="utf-8") as fout:
                for word in new_vocab:
                    fout.write('%s\n' % word)
            logging.info('lemmatization done and cached to file')
            print('lemmatization done and cached to file')

    def _get_top_words_dist(self, state, cutoff):
        #print(list(self.semfit.values()))
        distances = {}
        for k,v in self.semfit.items():
            distances[k] = scipy.spatial.distance.cosine(state,v)
        ks,vs = zip(*list(distances.items()))
        print('ks')
        print(ks)
        print('vs')
        print(vs) 
        top_k = []
        ap = list(np.argpartition(-1* np.asarray(vs,dtype=float),cutoff))[:cutoff]
        top_k = [ks[a] for a in ap]
        top_k_vals = np.asarray([vs[a] for a in ap],dtype= float)
        print(top_k_vals)
        
        print(top_k)
        probs = top_k_vals / top_k_vals.sum(axis=0)
        return top_k, probs

    def _embed_sentences(self, inst_id_to_sentence: Dict[str, Tuple[List[str], int]], disable_symmetric_patterns) -> \
            Tuple[List, List]:
        inst_id_sent_tuples = list(inst_id_to_sentence.items())
        target = inst_id_sent_tuples[0][0].rsplit('.', 1)[0]
        to_embed = []

        if disable_symmetric_patterns:
            # w/o sym. patterns - predict for blanked out word.
            # if the target word is the first or last in sentence get empty prediction by embedding '.'
            for _, (tokens, target_idx) in inst_id_sent_tuples:
                forward = tokens[:target_idx]
                backward = tokens[target_idx + 1:]
                if not forward:
                    forward = ['.']
                if not backward:
                    backward = ['.']
                to_embed.append(forward)
                to_embed.append(backward)
        else:

            # w/ sym. patterns - include target word + "and" afterwards in both directions
            for _, (tokens, target_idx) in inst_id_sent_tuples:
                # forward sentence
                to_embed.append(tokens[:target_idx + 1] + ['y']) #RL

                # backward sentence
                to_embed.append(['y'] + tokens[target_idx:]) #RL

        logging.info('embedding %d sentences for target %s' % (len(to_embed), target))
        embedded = self.elmo.sents2elmo(to_embed)
        #print(list([len(x)] for x in to_embed))
        return inst_id_sent_tuples, embedded

    

    def predict_sent_substitute_representatives(self, inst_id_to_sentence: Dict[str, Tuple[List[str], int]],
                                                n_represent: int,
                                                n_samples_side: int, disable_symmetric_patterns: bool,
                                                disable_lemmatiziation: bool, prediction_cutoff: int) \
            -> Dict[str, List[Dict[str, int]]]:
        """
        a representative is a dictionary made out of samples from both sides of the BiLM, predicting substitutes
        for a contextualized token.
        an example might look like:
        {'forward_jump':2,'backward_leap':1, 'backward_climb':1} (n_samples_side=2)
        we return a list of n_representatives of those

        :param inst_id_to_sentence: dictionary instance_id -> (sentence tokens list, target word index in tokens)
        :param n_represent: number of representatives
        :param n_samples_side: number of samples to draw from each side
        :param disable_symmetric_patterns: if true words are predicted from context only
        :param disable_lemmatiziation: if true predictions are not lemmatized
        :param prediction_cutoff: only top prediction_cutoff LM prediction are considered
        :return: map from instance id to list of representatives
        """
        inst_id_sent_tuples, embedded = self._embed_sentences(inst_id_to_sentence, disable_symmetric_patterns)
        lemma = inst_id_sent_tuples[0][0].split('.')[0]

        #vocabulary_used = self.semfit.keys()#self.elmo_word_vocab if disable_lemmatiziation else self.elmo_word_vocab_lemmatized

        #TIRO CODIGO
        forwards = []
        backwards = []
        for i, elem in enumerate(embedded):
            
            if i % 2 == 0:
                forwards.append(elem[:, :512])
            else:
                backwards.append(elem[:,512:])
        
        #--

        results = {}
        for i in range(len(inst_id_sent_tuples)):
            inst_id, (tokens, target_idx) = inst_id_sent_tuples[i]
            target_word_lower = tokens[target_idx].lower()

            sentence = ' '.join([t if i != target_idx else '***%s***' % t for i, t in enumerate(tokens)])
            logging.info('instance %s sentence: %s' % (inst_id, sentence))

            # these will be multiplied by ELMo's output matrix, [token-index, state dims]
            # (first 512 state dims in elmo are the forward LM, 512:1024 are the backward LM)
            #print(embedded)
            forward_out_em = forwards[i][-1,:] #embedded[i * 2][-1, :512] #[2, -1, :512]
            backward_out_em = backwards[i][0,:] #embedded[i * 2 + 1][0, 512:] #[2, 0, 512:]

            forward_idxs, forward_dist = self._get_top_words_dist(forward_out_em, prediction_cutoff)
            backward_idxs, backward_dist = self._get_top_words_dist(backward_out_em, prediction_cutoff)

            forward_samples = []

            # after removing samples equal to disamb. target,
            # we might end up with not enough samples, so repeat until we have enough samples
            while len(forward_samples) < n_represent * n_samples_side:
                new_samples = list(
                    np.random.choice(forward_idxs, n_represent * n_samples_side * 2,
                                     p=forward_dist))
                new_samples = [x for x in new_samples if x.lower() != lemma and x.lower() != target_word_lower]
                forward_samples += new_samples

            backward_samples = []
            while len(backward_samples) < n_represent * n_samples_side:
                new_samples = list(
                    np.random.choice(backward_idxs, n_represent * n_samples_side * 2,
                                     p=backward_dist))
                new_samples = [x for x in new_samples if x.lower() != lemma and x.lower() != target_word_lower]
                backward_samples += new_samples
            logging.info('some forward samples: %s' % [x for x in forward_samples[:5]])
            logging.info('some backward samples: %s' % [x for x in backward_samples[:5]])
            representatives = []
            for _ in range(n_represent):
                representative = dict()
                for _ in range(n_samples_side):
                    for sample_src in forward_samples, backward_samples:
                        sample_word = sample_src.pop()
                        representative[sample_word] = representative.get(sample_word, 0) + 1
                representatives.append(representative)
            logging.info('first 3 representatives out of %d:\n%s' % (n_represent, representatives[:3]))
            results[inst_id] = representatives
        return results