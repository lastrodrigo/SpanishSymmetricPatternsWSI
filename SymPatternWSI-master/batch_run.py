from spwsi_elmo import main
from spwsi.spwsi import DEFAULT_PARAMS
import argparse
import numpy as np

if __name__ == "__main__":
    n_runs = 100

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
    parser.add_argument('--elmo-vocab-path',dest='elmo_vocab_path', type=str,default='./resources/vocab.txt',
                        help='path to elmo training vocabulary file')
    parser.add_argument('--weights-path',dest='weights_path',type=str,
                        default='./resources/weights.hdf5',
                        help='path to elmo softmax weights')
    parser.add_argument('--task',dest='task',type=str, default='SENSEVAL_2_SLS',
                            help='Task selection, possible values: SE2SLS (default), SE2015T13, SE2013T3')
    parser.add_argument('--taskPath',dest='taskPath',type=str, help='path to task resources directory')
    parser.add_argument('--maxLabels',dest='maxLabels',type=int, default= '2', help='max number of labels per instance to generate key')
    parser.add_argument('--options-path', dest='optionsPath',type=str, help='Path to options file',default='./resources/options.json')
    #-
    args = parser.parse_args()
    
    precisions = []
    corrects = []
    attempteds = []
    recalls = []
    attempted_pcts = []
    for run in range(0, n_runs):
        print('Run N° %d' % (run + 1))
        print('------------')
        results = main(args)['all']
        precisions.append(results['precision'])
        corrects.append(results['correct'])
        attempteds.append(results['attempted'])
        recalls.append(results['recall'])
        attempted_pcts.append(results['attemptedPct'])
        total = results['total']
    print('Averages')
    print('--------')
    print('Precision: %f' % np.mean(precisions))
    print('Correct: %f' % np.mean(corrects))
    print('Attempted: %f' % np.mean(attempteds))
    print('Recall: %f' % np.mean(recalls))
    print('Attempted Pct: %f' % np.mean(attempted_pcts))
    print('Total: %d' % total)
