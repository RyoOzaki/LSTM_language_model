import numpy as np

def onehot(label_matrix, depth=None):
    shape = label_matrix.shape
    if depth is None:
        depth = int(label_matrix.max()+1)
    identity = np.identity(depth, dtype=np.float32)
    label_matrix_tmp = label_matrix.copy().astype(int)
    label_matrix_tmp[label_matrix < 0] = -1
    output = identity[label_matrix_tmp]
    for tup in zip(*np.where(label_matrix < 0)):
        output[tup] = label_matrix[tup]
    return output

def make_input_output(sentences):
    N, T = sentences.shape
    input = sentences[:, :-1].copy()
    output = sentences[:, 1:].copy()
    input[output < 0] = output[output < 0]

    return input, output
