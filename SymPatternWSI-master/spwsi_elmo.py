from spwsi.bilm_elmo import BilmElmo
import argparse
import os
import logging
from time import strftime
from spwsi.spwsi import DEFAULT_PARAMS, SPWSI, Task #RL added Task

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BiLM Symmetric Patterns WSI Demo',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n-clusters', dest='n_clusters', type=int, default=DEFAULT_PARAMS['n_clusters'],
                        help='number of clusters per instance')
    parser.add_argument('--n-representatives', dest='n_represent', type=int, default=DEFAULT_PARAMS['n_represent'],
                        help='number of representations per sentence')
    parser.add_argument('--n-samples-side', dest='n_samples_side', type=int, default=DEFAULT_PARAMS['n_samples_side'],
                        help='number of samples per representations side')
    parser.add_argument('--cuda_device', dest='cuda_device', type=int, default=DEFAULT_PARAMS['cuda_device'],
                        help='cuda device for ELMo (-1 to disable)')
    parser.add_argument('--debug-dir', dest='debug_dir', type=str, default=DEFAULT_PARAMS['debug_dir'],
                        help='logs and keys are written will be written to this dir')
    parser.add_argument('--disable-lemmatization', dest='disable_lemmatization',
                        default=DEFAULT_PARAMS['disable_lemmatization'], action='store_true',
                        help='disable ELMO prediction lemmatization')
    parser.add_argument('--disable-symmetric-patterns', dest='disable_symmetric_patterns',
                        default=DEFAULT_PARAMS['disable_symmetric_patterns'], action='store_true',
                        help='disable "x and y" symmetric pattern and predict substitutes inplace')
    parser.add_argument('--disable-tfidf', dest='disable_tfidf', action='store_true',
                        default=DEFAULT_PARAMS['disable_tfidf'],
                        help='disable tfidf transformer')
    parser.add_argument('--run-postfix', dest='run_postfix', type=str, default=DEFAULT_PARAMS['run_postfix'],
                        help='will be appended to log file names and products')
    parser.add_argument('--lm-batch-size', dest='lm_batch_size', type=int, default=DEFAULT_PARAMS['lm_batch_size'],
                        help='ELMo prediction batch size (optimization only)')
    parser.add_argument('--prediction-cutoff', dest='prediction_cutoff', type=int,
                        default=DEFAULT_PARAMS['prediction_cutoff'],
                        help='ELMo predicted distribution top K cutoff')
    parser.add_argument('--cutoff-lm-vocab', dest='cutoff_lm_vocab', type=int,
                        default=DEFAULT_PARAMS['cutoff_lm_vocab'],
                        help='optimization: only use top K words for faster output matrix multiplication')
    #+RL
    parser.add_argument('--elmo-vocab-path',dest='elmo_vocab_path', type=str,default='./resources/vocab-2016-09-10.txt',
                        help='path to elmo training vocabulary file')
    parser.add_argument('--weights-path',dest='weights_path',type=str,
                        default='./resources/elmo_2x4096_512_2048cnn_2xhighway_softmax_weights.hdf5',
                        help='path to elmo softmax weights')
    parser.add_argument('--task',dest='task',type=str, default='SENSEVAL_2_SLS',
                            help='Task selection, possible values: SE2SLS (default), SE2015T13, SE2013T3')
    parser.add_argument('--taskPath',dest='taskPath',type=str, help='path to task resources directory')
    parser.add_argument('--maxLabels',dest='maxLabels',type=int, default= '2', help='max number of labels per instance to generate key')
    #-
    args = parser.parse_args()

    startmsg = 'BiLM Symmetric Patterns WSI Demo\n\n'
    startmsg += 'Arguments:\n'
    startmsg += '-' * 10 + '\n'
    for arg in vars(args):
        startmsg += (' %-30s:%s\n' % (arg.replace('_', '-'), getattr(args, arg)))
    startmsg = startmsg.strip()
    print(startmsg)

    run_name = strftime("%m%d-%H%M%S") + ('-' + args.run_postfix if args.run_postfix else '')
    print('this run name: %s' % run_name)
    if args.debug_dir:
        if not os.path.exists(args.debug_dir):
            os.makedirs(args.debug_dir)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(args.debug_dir, '%s.log.txt' % run_name), 'w', 'utf-8')
        formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)  # Pass handler as a parameter, not assign
        root_logger.addHandler(handler)
    logging.info(startmsg)

    #RL elmo_vocab_path = './resources/vocab-2016-09-10.txt'
    BilmElmo.create_lemmatized_vocabulary_if_needed(args.elmo_vocab_path)
    elmo_as_lm = BilmElmo(args.cuda_device, args.weights_path, #RL
                          args.elmo_vocab_path,
                          batch_size=args.lm_batch_size,
                          cutoff_elmo_vocab=args.cutoff_lm_vocab)
    spwsi_runner = SPWSI(elmo_as_lm)

    scores = spwsi_runner.run(n_clusters=args.n_clusters, n_represent=args.n_represent,
                              n_samples_side=args.n_samples_side, disable_lemmatization=args.disable_lemmatization,
                              disable_tfidf=args.disable_tfidf,
                              disable_symmetric_patterns=args.disable_symmetric_patterns,
                              prediction_cutoff=args.prediction_cutoff,
                              debug_dir=args.debug_dir, run_name=run_name,
                              taskPath= args.taskPath, task= args.task, maxLabels= args.maxLabels, #RL added
                              print_progress=True)
    logging.info('full results: %s' % scores)
