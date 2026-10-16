"""CPU-only Keras hemolysis inference with the original custom attention layer."""

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import (
    Activation,
    Add,
    Concatenate,
    Dense,
    Dot,
    Lambda,
    Layer,
    RepeatVector,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences


class Attention(Layer):
    """Luong or Bahdanau attention compatible with the distributed HDF5 model."""

    def __init__(self, units=128, score="luong", **kwargs):
        super().__init__(**kwargs)
        if score not in ("luong", "bahdanau"):
            raise ValueError("Attention score must be 'luong' or 'bahdanau'.")
        self.units = units
        self.score = score

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        with backend.name_scope(self.name):
            if self.score == "luong":
                self.luong_w = Dense(input_dim, use_bias=False, name="luong_w")
                self.luong_dot = Dot(axes=[1, 2], name="attention_score")
            else:
                self.bahdanau_v = Dense(1, use_bias=False, name="bahdanau_v")
                self.bahdanau_w1 = Dense(input_dim, use_bias=False, name="bahdanau_w1")
                self.bahdanau_w2 = Dense(input_dim, use_bias=False, name="bahdanau_w2")
                self.bahdanau_repeat = RepeatVector(input_shape[1])
                self.bahdanau_tanh = Activation("tanh", name="bahdanau_tanh")
                self.bahdanau_add = Add()
            self.h_t = Lambda(
                lambda hidden: hidden[:, -1, :],
                output_shape=(input_dim,),
                name="last_hidden_state",
            )
            self.softmax_normalizer = Activation("softmax", name="attention_weight")
            self.dot_context = Dot(axes=[1, 1], name="context_vector")
            self.concat_c_h = Concatenate(name="attention_output")
            self.w_c = Dense(self.units, use_bias=False, activation="tanh", name="attention_vector")
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def call(self, inputs, training=None, **kwargs):
        last_hidden = self.h_t(inputs)
        if self.score == "luong":
            score = self.luong_dot([last_hidden, self.luong_w(inputs)])
        else:
            first = self.bahdanau_repeat(self.bahdanau_w1(last_hidden))
            second = self.bahdanau_w2(inputs)
            score = self.bahdanau_v(self.bahdanau_tanh(self.bahdanau_add([first, second])))
            score = backend.squeeze(score, axis=-1)
        weights = self.softmax_normalizer(score)
        context = self.dot_context([inputs, weights])
        return self.w_c(self.concat_c_h([context, last_hidden]))

    def get_config(self):
        return {**super().get_config(), "units": self.units, "score": self.score}


class Hemolysis:
    """Return P(non-hemolytic), using left padding to 200 residues."""

    def __init__(self, path):
        with tf.device("/CPU:0"):
            self.model = load_model(path, custom_objects={"Attention": Attention}, compile=False)

    def predict(self, sequences: list[str]) -> list[float]:
        vocabulary = "XACDEFGHIKLMNPQRSTVWY"
        encoded = [[vocabulary.index(amino_acid) for amino_acid in seq] for seq in sequences]
        padded = pad_sequences(encoded, maxlen=200)
        with tf.device("/CPU:0"):
            return self.model(padded, training=False).numpy().reshape(-1).tolist()
