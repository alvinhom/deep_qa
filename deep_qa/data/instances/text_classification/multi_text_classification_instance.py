from typing import Dict, List

import numpy
from overrides import overrides

from ..instance import TextInstance, IndexedInstance
from ...data_indexer import DataIndexer


class MultipleLabelTextClassificationInstance(TextInstance):
    """
    A MultipleLabelTextClassificationInstance is a :class:`TextInstance` that is a single passage of text,
    where that passage has some associated (multiple categorical) label.
    """
    def __init__(self, text: str, label: bool, index: int=None):
        """
        text: the text of this instance, typically either a sentence or a logical form.
        """
        super(MultipleLabelTextClassificationInstance, self).__init__(label, index)
        self.text = text

    def __str__(self):
        return 'MultipleLabelTextClassificationInstance(' + self.text + ', ' + str(self.label) + ')'

    @overrides
    def words(self) -> Dict[str, List[str]]:
        return self._words_from_text(self.text)

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        indices = self._index_text(self.text, data_indexer)
        return IndexedMultipleLabelTextClassificationInstance(indices, self.label, self.index)

    @classmethod
    def read_from_line(cls, line: str, default_label: bool=None):
        """
        Reads a MultipleLabelTextClassificationInstance object from a line.  The format has one of four options:
        (1) [sentence][tab][label]
        """
        fields = line.split("\t")

        # We'll call Instance._check_label for all four cases, even though it means passing None to
        # two of them.  We do this mainly for consistency, and in case the _check_label() ever
        # changes to actually do something with the label=None case.
        if len(fields) == 2:
            text, label_string = fields
            label = label_string
            return cls(text, label)
        elif len(fields) == 1:
            text = fields[0]
            return cls(text, default_label)
        else:
            raise RuntimeError("Unrecognized line format: " + line)


class IndexedMultipleLabelTextClassificationInstance(IndexedInstance):
    def __init__(self, word_indices: List[int], label, index: int=None):
        super(IndexedMultipleLabelTextClassificationInstance, self).__init__(label, index)
        self.word_indices = word_indices

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedMultipleLabelTextClassificationInstance([], label=None, index=None)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        """
        This simple IndexedInstance only has one padding dimension: word_indices.
        """
        return self._get_word_sequence_lengths(self.word_indices)

    @overrides
    def pad(self, padding_lengths: Dict[str, int]):
        self.word_indices = self.pad_word_sequence(self.word_indices, padding_lengths)

    @overrides
    def as_training_data(self):
        word_array = numpy.asarray(self.word_indices, dtype='int32')
        label = self.label
        """
        Note, the label must be converted into a categorical int later.  Do it in the model
        """
        return word_array, label
