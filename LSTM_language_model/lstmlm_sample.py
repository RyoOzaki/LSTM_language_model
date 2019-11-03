import numpy as np
from model.LSTM_language_model import LSTM_language_model
from util.utility import onehot, state2word, sentences2strings, make_input_output
from pathlib import Path

npz_obj = np.load("weathernews.npz")
N, T = npz_obj["shape"]
D = npz_obj["depth"]
sentence_matrix = npz_obj["sentences"]
input, output = make_input_output(sentence_matrix)
input_matrix = onehot(input, depth=D)
output_matrix = onehot(output, depth=D)
words = list(npz_obj["words"])

BOS = "<BOS>"
EOS = "<EOS>"

BOS_index = words.index(BOS)
EOS_index = words.index(EOS)

batch_size = 128
epochs = 100
train_times = 10
latent_dim = 128

lstmlm = LSTM_language_model(D, latent_dim)

result_dir = Path("result")
result_dir.mkdir(exist_ok=True)

for tr in range(train_times):
    lstmlm.fit(
        input_matrix, output_matrix,
        batch_size=batch_size, epochs=epochs,
        validation_split=0.1
    )
    sentences = sentences2strings(state2word(lstmlm.generate(size=100), words))

    rslt_dir = result_dir / f"ITER_{int((tr + 1) * epochs)}"
    rslt_dir.mkdir(exist_ok=True)

    lstmlm.save_model(rslt_dir / "model.h5")
    snt_out_file = rslt_dir / "sentences.txt"
    snt_out_file.write_text("\n".join(sentences))
