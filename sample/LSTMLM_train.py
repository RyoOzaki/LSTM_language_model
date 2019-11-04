import numpy as np
from LSTM_language_model.model import LSTM_language_model
from LSTM_language_model.utility import onehot, make_input_output
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--latent_node", type=int, required=True)
parser.add_argument("--epoch", type=int, required=True)
parser.add_argument("--continue_train", action="store_true")

args = parser.parse_args()

npz_obj = np.load(args.dataset)
N, T = npz_obj["shape"]
D = npz_obj["depth"]
sentence_matrix = npz_obj["sentences"]
input, output = make_input_output(sentence_matrix)
input_matrix = onehot(input, depth=D)
output_matrix = onehot(output, depth=D)
words = list(npz_obj["words"])

BOS_index = 0
EOS_index = D-1

batch_size = args.batch_size
epochs = args.epoch
latent_node = args.latent_node

args.model.parent.mkdir(exist_ok=True, parents=True)

if args.continue_train:
    lstmlm = LSTM_language_model(D, None, BOS_index=BOS_index, EOS_index=EOS_index, load_model_path=args.model)
else:
    lstmlm = LSTM_language_model(D, latent_node, BOS_index=BOS_index, EOS_index=EOS_index)

lstmlm.fit(
    input_matrix, output_matrix,
    batch_size=batch_size, epochs=epochs,
    # validation_split=0.1
)
lstmlm.save_model(args.model)
