from typing import List, Dict, Tuple
import logging
from allennlp.commands.elmo import ElmoEmbedder
import h5py
import numpy as np
import spacy
import os
from xml.etree import ElementTree
from collections import defaultdict

class EmbeddingBaseline:

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
            _ = list(self.elmo.embed_sentences([warm_up_sent] * self.batch_size, self.batch_size))

    def __init__(self, cuda_device, weights_path, options_path, batch_size=40,
                 cutoff_elmo_vocab=50000):
        super().__init__()
        logging.info(
            'creating elmo in device %d. weight path %s '
            ' batch_size: %d' % (
                cuda_device, weights_path,
                batch_size))
        self.elmo = ElmoEmbedder(cuda_device=cuda_device, weight_file= weights_path, options_file=options_path )

        self.batch_size = batch_size

        logging.info('warming up elmo')
        self._warm_up_elmo()

        logging.info('reading elmo weights')
        with h5py.File(weights_path, 'r', libver='latest', swmr=True) as fin:
            self.elmo_softmax_w = fin['softmax/W'][:cutoff_elmo_vocab, :].transpose()


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
        embedded = list(self.elmo.embed_sentences(to_embed, self.batch_size))

        return inst_id_sent_tuples, embedded

def replace_acuted(word: str): #+RL
    return word.replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')

def generate_senseval_2(dir_path: str, dictionary: dict): #+RL
    logging.info('reading Senseval dataset from %s' % dir_path)
    nlp = spacy.load("es", disable=['ner','parser'])
    in_xml_path = os.path.join(dir_path,'test/test.xml')
    gold_key_path = os.path.join(dir_path,'key')

    with open(in_xml_path,encoding='ISO-8859-1') as fin_xml, open(gold_key_path, encoding="utf8") as fin_key:
        instid_in_key = set()
        lemmas = dict()
        for line in fin_key:
            lemma_pos, inst_id, _ = line.strip().split(maxsplit=2)
            if not (lemma_pos in lemmas.keys()):
                pos = dictionary[lemma_pos]['pos']
                lemmas[lemma_pos] = pos
                break
            instid_in_key.add(inst_id)
        et_xml = ElementTree.parse(fin_xml)
        for word in et_xml.getroot():
            for inst in word.getchildren():
                inst_id = inst.attrib['id']
                if inst_id not in instid_in_key:
                    continue
                lemma_pos = inst_id.split('.')[0]
                lemma_pos += '.' + lemmas[lemma_pos] 
                context = inst.find("context")
                before, target, after = list(context.itertext())
                before = [x.text for x in nlp(before.strip(),disable=['parser','tagger','ner'])]
                target = target.strip()
                after = [x.text for x in nlp(after.strip(), disable=['parser','tagger','ner'])]
                yield before + [target] + after, inst_id, lemma_pos

def get_senseval2_dictionary(dir_path:str):
    dict_path = os.path.join(dir_path,'test/senseval.dict')
    dictionary = {}
    with open(dict_path,encoding='ISO-8859-1') as fin_dict:
        for line in fin_dict:
            splitted = line.split('#')
            word = replace_acuted(splitted[0]) 
            if word not in ['','\n']:
                if word[-1] == 's':
                    word = word[:-1]
                pos = splitted[1][0].lower()
                if pos == 'a':
                    pos= 'j'
                if not word in dictionary:
                    dictionary[word] = {'pos': pos, 'senses': {}}
                dictionary[word]['senses'][splitted[2]] = (splitted[3], None)
    return dictionary

def __main__():
    weights_path = '../SymPatternWSI-master/resources/weights.hdf5'
    options_path = '../SymPatternWSI-master/resources/options.json'
    taskPath = '../SymPatternWSI-master/spanish-lex-sample'
    semeval_dataset_by_target = defaultdict(dict)
    embedding_baseline = EmbeddingBaseline(cuda_device= 0, weights_path=weights_path, options_path=options_path)
    semeval_dictionary = get_senseval2_dictionary(taskPath)

    for tokens, inst_id, lemma_pos in generate_senseval_2(taskPath, semeval_dictionary):
        semeval_dataset_by_target[lemma_pos][inst_id] = tokens
