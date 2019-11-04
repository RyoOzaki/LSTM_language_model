import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Masking
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam

class LSTM_language_model(object):

    __MASKING_VALUE = -1

    def __init__(self, input_dim, hidden_dim, BOS_index=0, EOS_index=-1, load_model_path=None):

        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        if EOS_index < 0:
            EOS_index += input_dim
        self.__BOS_index = BOS_index
        self.__EOS_index = EOS_index

        if load_model_path:
            model = load_model(load_model_path)
        else:
            model = self.__define_model(input_dim, hidden_dim)
        self.__model = model

    def __define_model(self, input_dim, hidden_dim):

        model = Sequential()
        model.add(Masking(input_shape=(None, input_dim), mask_value=LSTM_language_model.__MASKING_VALUE))
        model.add(LSTM(hidden_dim, return_sequences=True))
        model.add(TimeDistributed(Dense(input_dim, activation="softmax")))

        model.compile(loss="categorical_crossentropy", optimizer=Adam())

        return model

    @property
    def model(self):
        return self.__model

    def generate(self, size=1, Tmax=None):
        generated_states = []
        n_range = np.arange(size)
        input_matrix = np.zeros((size, 1, self.__input_dim), dtype=np.float32)
        input_matrix[:, 0, self.__BOS_index] = 1
        for n in range(size):
            generated_states.append([self.__BOS_index])
        t = 1
        while True:
            t += 1
            v = self.model.predict(input_matrix)[:, -1].reshape((-1, self.__input_dim))
            sampled_v = np.zeros(v.shape[0], dtype=int)
            for i, (n, prob) in enumerate(zip(n_range, v)):
                sampled_v[i] = np.random.choice(self.__input_dim, p=prob)
                generated_states[n].append(sampled_v[i])
            unfinished_flag = (sampled_v != self.__EOS_index)
            unfinished_size = unfinished_flag.sum()
            if unfinished_size == 0:
                break
            elif Tmax is not None and t >= Tmax:
                break
            n_range = n_range[unfinished_flag]
            unfinished_input = input_matrix[unfinished_flag].reshape((unfinished_size, -1, self.__input_dim))
            input_matrix = np.concatenate((unfinished_input, np.zeros((unfinished_size, 1, self.__input_dim), dtype=np.float32)), axis=1)
            for i, idx in enumerate(sampled_v[unfinished_flag]):
                input_matrix[i, -1, idx] = 1
        return generated_states

    def save_model(self, path, **kwargs):
        self.__model.save(path, **kwargs)

    def fit(self, *argv, **kwargs):
        self.__model.fit(*argv, **kwargs)
