import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epochs', default=2, type=int, help='Number of epochs to train')
    parser.add_argument('--patience', default=40, type=int, help='Patience for early stopping')
    parser.add_argument('--epoch_len', default=None, type=int, help='number of iterations in an epoch')
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size for the randomly sampled batch')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Learning rate')

    # Non tunable flags
    parser.add_argument('--expt_dir', default='./results', type=str, help='The name of the experiment dir')
    parser.add_argument('--dataset', default='weather', type=str, help='The name of the dataset.')
    parser.add_argument('--datetime_col', default='date', type=str, help='Column having datetime.')
    parser.add_argument('--num_cov_cols', default=None, type=list, help='Column having numerical features.')
    parser.add_argument('--cat_cov_cols', default=None, type=list, help='Column having categorical features.')
    parser.add_argument('--hist_len', default=720, type=int, help='Length of the history provided as input')
    parser.add_argument('--pred_len', default=96, type=int, help='Length of pred len during training')
    parser.add_argument('--num_layers', default=1, type=int, help='Number of DNN layers')
    parser.add_argument('--hidden_size', default=256, type=int, help='Hidden size of DNN')
    parser.add_argument('--decoder_output_dim', default=8, type=int, help='Hidden d3 of DNN')
    parser.add_argument('--final_decoder_hidden', default=64, type=int, help='Hidden d3 of DNN')
    parser.add_argument('--ts_cols', default=None, type=list, help='Columns of time-series features')
    parser.add_argument(
        '--random_seed', default=None, type=int, help='The random seed to be used for TF and numpy'
    )
    parser.add_argument('--normalize', default=True, type=bool, help='normalize data for training or not')
    parser.add_argument('--holiday', default=False, type=bool, help='use holiday features or not')
    parser.add_argument('--permute', default=True, type=bool, help='permute the order of TS in training set')
    parser.add_argument('--transform', default=False, type=bool, help='Apply chronoml transform or not.')
    parser.add_argument('--layer_norm', default=True, type=bool, help='Apply layer norm or not.')
    parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--num_split', default=1, type=int, help='number of splits during inference.')
    parser.add_argument(
        '--min_num_epochs', default=0, type=int, help='minimum number of epochs before early stopping'
    )
    parser.add_argument('--gpu', default=0, type=int, help='index of gpu to be used.')

    return parser.parse_args()
