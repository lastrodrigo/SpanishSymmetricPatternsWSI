import logging
from elmoformanylangs import Embedder
from .bilm_interface import Bilm

class BilmElmo(Bilm) :

    def __init__(self, cuda_device, model_path, vocab_path, batch_size=64,
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

