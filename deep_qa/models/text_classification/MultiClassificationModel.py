from overrides import overrides

import numpy
import dill as pickle
from keras.layers import Dense, Dropout, Input
from keras.utils.np_utils import to_categorical
from typing import Dict, List

from ...data.instances.text_classification.multi_text_classification_instance import MultipleLabelTextClassificationInstance
from ...training.text_trainer import TextTrainer
from ...training.models import DeepQaModel
from ...common.params import Params
from ...data.dataset import IndexedDataset
from ...data.data_indexer import DataIndexer

class MultiClassificationModel(TextTrainer):
    """
    A TextTrainer that simply takes word sequences as input (could be either sentences or logical
    forms), encodes the sequence using a sentence encoder, then uses a few dense layers to decide
    on some classification label for the text sequence.  This class support multi class
    classification versus the default ClassficationModel
    """
    def __init__(self, params: Params):
        self.label_data_indexer = DataIndexer()
        self.num_classes = params.pop('num_classes', None)
        super(MultiClassificationModel, self).__init__(params)

    @overrides
    def _build_model(self):
        '''
        train_input: numpy array: int32 (samples, num_words). Left padded arrays of word indices
            from sentences in training data
        '''
        # Step 1: Convert the sentence input into sequences of word vectors.
        sentence_input = Input(shape=self._get_sentence_shape(), dtype='int32', name="sentence_input")
        word_embeddings = self._embed_input(sentence_input)

        # Step 2: Pass the sequences of word vectors through the sentence encoder to get a sentence
        # vector..
        sentence_encoder = self._get_encoder()
        sentence_encoding = sentence_encoder(word_embeddings)

        # Add a dropout after LSTM.
        regularized_sentence_encoding = Dropout(0.2)(sentence_encoding)

        # Step 3: Find p(true | proposition) by passing the outputs from LSTM through an MLP with
        # ReLU layers.
        projection_layer = Dense(int(self.embedding_dim['words']/2), activation='relu', name='projector')
        # We need a better way to determine the number of classes from the data?
        softmax_layer = Dense(self.num_classes, activation='softmax', name='softmax')
        output_probabilities = softmax_layer(projection_layer(regularized_sentence_encoding))

        # Step 4: Define crossentropy against labels as the loss.
        return DeepQaModel(inputs=sentence_input, outputs=output_probabilities)

    def _instance_type(self):
        return MultipleLabelTextClassificationInstance

    @overrides
    def _set_padding_lengths_from_model(self):
        self._set_text_lengths_from_model_input(self.model.get_input_shape_at(0)[1:])

    @overrides
    def get_padding_memory_scaling(self, padding_lengths: Dict[str, int]) -> int:
         return padding_lengths['num_sentence_words'] ** 2

    @overrides
    def get_instance_sorting_keys(self) -> List[str]:  # pylint: disable=no-self-use
         return ['num_sentence_words']

    @overrides
    def create_data_arrays(self, dataset: IndexedDataset):
        """
        Need to convert labels from string to indices
        """
        inputs, labels = super(MultiClassificationModel, self).create_data_arrays(dataset)
        label_indices = []
        # Convert labels to indices
        for label in labels:
            index = self.label_data_indexer.add_word_to_index(label)
            label_indices.append(index)
        return inputs, to_categorical(numpy.asarray(label_indices))

    @overrides
    def _save_auxiliary_files(self):
        super(MultiClassificationModel, self)._save_auxiliary_files()
        label_data_indexer_file = open("%s_label_data_indexer.pkl" % self.model_prefix, "wb")
        pickle.dump(self.label_data_indexer, label_data_indexer_file)
        label_data_indexer_file.close()

    @overrides
    def _load_auxiliary_files(self):
        super(MultiClassificationModel, self)._load_auxiliary_files()
        label_data_indexer_file = open("%s_label_data_indexer.pkl" % self.model_prefix, "rb")
        self.label_data_indexer = pickle.load(label_data_indexer_file)
        label_data_indexer_file.close()

    def get_label_text(self, index):
        # indexer by default has 0=padding 1=unknown, so need to increase by 2
        return self.label_data_indexer.get_word_from_index(index)
