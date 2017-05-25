from overrides import overrides

import numpy
import dill as pickle
import logging
from keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten
from keras.utils.np_utils import to_categorical
from typing import Dict, List

from ...data.instances.text_classification.multi_text_classification_instance import MultipleLabelTextClassificationInstance
from ...training.text_trainer import TextTrainer
from ...training.models import DeepQaModel
from ...common.params import Params
from ...data.dataset import IndexedDataset
from ...data.data_indexer import DataIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

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
        self.num_hidden_layers = params.pop('num_hidden_layers', 1)
        super(MultiClassificationModel, self).__init__(params)

    @overrides
    def _build_model(self):
        '''
        train_input: numpy array: int32 (samples, num_words). Left padded arrays of word indices
            from sentences in training data
        '''
        # Step 1: Convert the sentence input into sequences of word vectors.
        sentence_input = Input(shape=self._get_sentence_shape(), dtype='int32', name="sentence_input")
        logger.info("Shape for sentence_input" + str(sentence_input))
        word_embeddings = self._embed_input(sentence_input)

        # Step 2: Pass the sequences of word vectors through the sentence encoder to get a sentence
        # vector..
        sentence_encoder = self._get_encoder()
        sentence_encoding = sentence_encoder(word_embeddings)

        # Add a dropout after LSTM.
        regularized_sentence_encoding = Dropout(0.2)(sentence_encoding)

        # Step 3: Find p(true | proposition) by passing the outputs from LSTM through an MLP with
        # ReLU layers.
        projection_layer = regularized_sentence_encoding
        for i in range(self.num_hidden_layers):
            projection_layer = Dense(int(self.embedding_dim['words']/2), activation='relu', name='hiddenlayer_%d' % i)(projection_layer)
        # We need a better way to determine the number of classes from the data?
        softmax_layer = Dense(self.num_classes, activation='softmax', name='softmax')
        output_probabilities = softmax_layer(projection_layer)

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
        def converterFn(trainingData):
            inputs, labels = trainingData
            label_indices = []
            # Convert labels to indices
            for label in labels:
                index = self.label_data_indexer.add_word_to_index(label)
                label_indices.append(index)
            return inputs, to_categorical(numpy.asarray(label_indices), self.label_data_indexer.get_vocab_size())
        if self.data_generator is not None:
            # Need to read in all the labels first as the generator split into batches
            instances = dataset.instances
            for instance in instances:
                self.label_data_indexer.add_word_to_index(instance.label)
            return self.data_generator.create_generator(dataset, converterFn)
        else:
            dataset.pad_instances(self.get_padding_lengths())
            data = dataset.as_training_data()
            return converterFn(data)

    @overrides
    def get_padding_memory_scaling(self, padding_lengths: Dict[str, int]) -> int:
        return padding_lengths['num_sentence_words'] ** 2
        
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
