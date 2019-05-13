from typing import List, Dict, Tuple
import logging
from allennlp.commands.elmo import ElmoEmbedder
import h5py
import numpy as np
import spacy
from tqdm import tqdm
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

    def __init__(self, cuda_device, weights_path, options_path, batch_size=40):
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



    def embed_sentences(self, inst_id_to_sentence):
        inst_id_sent_tuples = list(inst_id_to_sentence.items())
        target = inst_id_sent_tuples[0][0].rsplit('.', 1)[0]
        to_embed = []
        
        for _, (tokens,_) in inst_id_sent_tuples:
            
            to_embed.append(tokens)

        logging.info('embedding %d sentences for target %s' % (len(to_embed), target))
        embedded = list(self.elmo.embed_sentences(to_embed, self.batch_size))
        instance_embedding = dict()

        for index, (inst_id,_) in enumerate(inst_id_sent_tuples):

            instance_embedding[inst_id] = embedded[index][2][inst_id_to_sentence[inst_id][1]]
        return instance_embedding 

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
                
                before = [x.text for x in nlp(before.encode(encoding='ISO-8859-1').decode(encoding='UTF-8').strip(),disable=['parser','tagger','ner'])]
                target = target.encode(encoding='ISO-8859-1').decode(encoding='UTF-8').strip()
                after = [x.text for x in nlp(after.encode(encoding='ISO-8859-1').decode(encoding='UTF-8').strip(), disable=['parser','tagger','ner'])]

                yield before + [target] + after, len(before), inst_id, lemma_pos

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
                definition = splitted[3]
                
                dictionary[word]['senses'][splitted[2]] = (definition, None)
    return dictionary

def __main__():
    weights_path = '../SymPatternWSI-master/resources/weights.hdf5'
    options_path = '../SymPatternWSI-master/resources/options.json'
    taskPath = '../SymPatternWSI-master/spanish-lex-sample'
    print_progress = True
    semeval_dataset_by_target = defaultdict(dict)
    embedding_baseline = EmbeddingBaseline(cuda_device= 0, weights_path=weights_path, options_path=options_path,
                                            batch_size=20)
    semeval_dictionary = get_senseval2_dictionary(taskPath)
    nlp = spacy.load("es", disable=['ner','parser'])
    for tokens, target_idx, inst_id, lemma_pos in generate_senseval_2(taskPath, semeval_dictionary):

        semeval_dataset_by_target[lemma_pos][inst_id] = (tokens,target_idx)

    inst_id_to_sense = {}
    gen = semeval_dataset_by_target.items()
    
    if print_progress:
        gen = tqdm(gen, desc='embedding sentences')
    
    for lemma_pos, inst_id_to_sentence in gen:
        lemma = lemma_pos.split('.')[0]
        inst_ids_to_embeddings = embedding_baseline.embed_sentences(
            inst_id_to_sentence)

        for sense, (definition, _) in semeval_dictionary[lemma]['senses'].items():
            splitted_definition = definition.split(':')
            gloss = splitted_definition[0]
            examples = []
            tokenized_examples = []
            if len(splitted_definition) > 1:
                examples = [x for x in splitted_definition[1].split(';')]
                for example in examples:
                    tokenized_examples.append([x.text for x in nlp(example.strip(),disable=['parser','tagger','ner'])])
            gloss = [x.text for x in nlp(gloss,disable=['parser','tagger','ner'])]
            to_embed = [gloss] + tokenized_examples

            embedded = list(embedding_baseline.elmo.embed_sentences(to_embed, embedding_baseline.batch_size))
            for emb in embedded:
                layer_2_emb = emb[2]
                
                for e in layer_2_emb:
                    centroid = np.mean(e,axis=1, dtype=np.float64)
                    print(len(centroid))
        break

        