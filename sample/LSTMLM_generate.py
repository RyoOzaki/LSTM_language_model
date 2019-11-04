import numpy as np
from LSTM_language_model.model import LSTM_language_model
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--dataset", type=Path, required=True)

parser.add_argument("--separator", default=" ")
parser.add_argument("--format", default="word", choices=["state", "word"])
parser.add_argument("--offset", type=int, default=0)
parser.add_argument("--size", type=int, default=1)
parser.add_argument("--without_BOS", action="store_true")
parser.add_argument("--without_EOS", action="store_true")

args = parser.parse_args()

npz_obj = np.load(args.dataset)
D = npz_obj["depth"]
words = list(npz_obj["words"])

BOS_index = 0
EOS_index = D-1

lstmlm = LSTM_language_model(D, None, BOS_index=BOS_index, EOS_index=EOS_index, load_model_path=args.model)
sentences = lstmlm.generate(size=args.size)
if args.without_BOS:
    sentences = [snt[1:] for snt in sentences]
if args.without_EOS:
    sentences = [snt[:-1] for snt in sentences]

if args.format == "word":
    sentences = [[words[idx] for idx in snt] for snt in sentences]
elif args.format == "state":
    sentences = [[idx+args.offset for idx in snt] for snt in sentences]
sentences = [args.separator.join(map(str, snt)) for snt in sentences]

for snt in sentences:
    print(snt)
