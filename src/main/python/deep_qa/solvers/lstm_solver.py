from typing import Any, Dict
from overrides import overrides

from keras.layers import Dense, Dropout
from keras.models import Model

from ..data.text_instance import TrueFalseInstance
from ..data.dataset import TextDataset, IndexedDataset  # pylint: disable=unused-import
from .nn_solver import NNSolver


class LSTMSolver(NNSolver):
    """
    An NNSolver that simply takes word sequences as input (could be either sentences or logical
    forms), encodes the sequence using an LSTM, then uses a few dense layers to decide if the
    sentence encoding is true or false.

    We don't really expect this model to work.  The best it can do is basically to learn word
    cooccurrence information, similar to how the Salience solver works, and I'm not at all
    confident that this does that job better than Salience.  We've implemented this mostly as a
    simple baseline.
    """
    def __init__(self, params: Dict[str, Any]):
        super(LSTMSolver, self).__init__(params)

    @overrides
    def _build_model(self):
        '''
        train_input: numpy array: int32 (samples, num_words). Left padded arrays of word indices
            from sentences in training data
        '''
        # Step 1: Convert the sentence input into sequences of word vectors.
        input_layer, word_embeddings = self._get_embedded_sentence_input(
                input_shape=(self.max_sentence_length,), name_prefix='sentence')

        # Step 2: Pass the sequences of word vectors through the sentence encoder to get a sentence
        # vector..
        sentence_encoder = self._get_sentence_encoder()
        sentence_encoding = sentence_encoder(word_embeddings)

        # Add a dropout after LSTM.
        regularized_sentence_encoding = Dropout(0.2)(sentence_encoding)

        # Step 3: Find p(true | proposition) by passing the outputs from LSTM through an MLP with
        # ReLU layers.
        projection_layer = Dense(int(self.embedding_size/2), activation='relu', name='projector')
        softmax_layer = Dense(2, activation='softmax', name='softmax')
        output_probabilities = softmax_layer(projection_layer(regularized_sentence_encoding))

        # Step 4: Define crossentropy against labels as the loss.
        model = Model(input=input_layer, output=output_probabilities)
        return model

    def _instance_type(self):
        return TrueFalseInstance

    @overrides
    def _get_max_lengths(self) -> Dict[str, int]:
        return {'word_sequence_length': self.max_sentence_length}

    @overrides
    def _set_max_lengths(self, max_lengths: Dict[str, int]):
        self.max_sentence_length = max_lengths['word_sequence_length']

    @overrides
    def _set_max_lengths_from_model(self):
        self.max_sentence_length = self.model.get_input_shape_at(0)[1]