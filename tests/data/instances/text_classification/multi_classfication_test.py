# pylint: disable=no-self-use,invalid-name
import numpy

# pylint: disable=line-too-long
from deep_qa.data.instances.text_classification.multi_text_classification_instance import IndexedMultipleLabelTextClassificationInstance
from deep_qa.data.instances.text_classification.multi_text_classification_instance import MultipleLabelTextClassificationInstance
# pylint: enable=line-too-long
from ....common.test_case import DeepQaTestCase


class TestMultipleLabelTextClassificationInstance:
    @staticmethod
    def instance_to_line(text, label=None, index=None):
        line = ''
        if index is not None:
            line += str(index) + '\t'
        line += text
        if label is not None:
            line += '\t' + label
        return line

    def test_read_from_line_handles_one_column(self):
        text = "this is a sentence"
        instance = MultipleLabelTextClassificationInstance.read_from_line(text)
        assert instance.text == text
        assert instance.label is None
        assert instance.index is None

    def test_read_from_line_handles_two_column(self):
        text = 'this is a sentence'
        text1 = text + '\tone'
        instance = MultipleLabelTextClassificationInstance.read_from_line(text1)
        assert instance.text == text
        assert instance.label == 'one'
        assert instance.index is None

    def test_read_from_line_handles_two_column_with_label(self):
        index = None
        text = "this is a sentence"
        label = "one"
        line = self.instance_to_line(text, label, index)

        instance = MultipleLabelTextClassificationInstance.read_from_line(line)
        assert instance.text == text
        assert instance.label == label
        assert instance.index == index
