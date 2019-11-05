import numpy as np
from pathlib import Path
from functools import reduce
from argparse import ArgumentParser

BOS = "<BOS>"
EOS = "<EOS>"

parser = ArgumentParser()

parser.add_argument("--source", type=Path, required=True, help="source file (txt)")
parser.add_argument("--output", type=Path, help="if --output option is not specified, --output will parse as --source.with_suffix('.npz')")
parser.add_argument("--repeat", type=int, default=1, help="repeat time")
parser.add_argument("--separator", default=" ", help="separator of words")
parser.add_argument("--mask_value", type=int, default=-1, help="value of masking, it will use in fill up the short length sentence")

parser.add_argument("--format", default="word", choices=["word", "state"], help="format of source file, --format option will specify 'word' or 'state'")
parser.add_argument("--comment_header", default="#")
parser.add_argument("--BOS", default=BOS, help="string of BOS flag")
parser.add_argument("--EOS", default=EOS, help="string of EOS flag")

args = parser.parse_args()

lines = Path(args.source).read_text().split("\n")
lines = [l.split(args.separator) for l in lines if l and not l.startswith(args.comment_header)]

if args.format == "word":
    words = set()
    for line in lines:
        words |= set(line)
        line.insert(0, args.BOS)
        line.append(args.EOS)
    words = sorted(list(words))
    words.insert(0, args.BOS)
    words.append(args.EOS)
    word_N = len(words)
    state_sentence = [[words.index(wrd) for wrd in sentence] for sentence in lines] # BOS-> 0, EOS->-1 (like as array access in python)
elif args.format == "state":
    state_sentence = [[int(wrd)+1 for wrd in sentence] for sentence in lines]
    word_N = max([max(sentence) for sentence in state_sentence]) # because the value was incremented in state_sentence_tmp
    word_N += 2 # BOS & EOS
    for sentence in state_sentence:
        sentence.insert(0, 0)
        sentence.append(word_N-1)
    words = np.arange(word_N)

state_sentence = state_sentence * args.repeat

sentence_N = len(state_sentence)
max_len = reduce(max, [len(sentence) for sentence in state_sentence])

sentence_matrix = np.zeros((sentence_N, max_len), dtype=int)

for i, sentence in enumerate(state_sentence):
    for t, state in enumerate(sentence):
        sentence_matrix[i, t] = state
    sentence_matrix[i, t+1:] = args.mask_value

outfile = args.output or args.source.with_suffix(".npz")
np.savez(outfile,
    shape=sentence_matrix.shape,
    depth=word_N,
    words=words,
    sentences=sentence_matrix
)
