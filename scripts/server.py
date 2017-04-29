#!flask/bin/python
from flask import Flask, jsonify, request
import logging
import os
import shutil
import sys
import numpy
import operator
from pyhocon import ConfigFactory

# These have to be before we do any import from keras.  It would be nice to be able to pass in a
# value for this, but that makes argument passing a whole lot more complicated.  If/when we change
# how arguments work (e.g., loading a file), then we can think about setting this as a parameter.
import random
random.seed(13370)
numpy.random.seed(1337)  # pylint: disable=no-member

from keras import backend as K

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from deep_qa.common.checks import ensure_pythonhashseed_set
from deep_qa.common.params import Params, replace_none
from deep_qa.models import concrete_models
from deep_qa.models.reading_comprehension.bidirectional_attention import BidirectionalAttentionFlow
from deep_qa.models.text_classification import MultiClassificationModel

from deep_qa.data.dataset import TextDataset
from deep_qa.data.instances.reading_comprehension import CharacterSpanInstance
from deep_qa.data.instances.text_classification import TextClassificationInstance
from deep_qa.data.instances.text_classification import MultipleLabelTextClassificationInstance
from deep_qa.data.instances.multiple_choice_qa import MultipleTrueFalseInstance
from deep_qa.data.instances.multiple_choice_qa import QuestionAnswerInstance

# Enums
UNDEFINED = 0
TRUE_FALSE = "TrueFalse"
MULTIPLE_TRUE_FALSE = "MultipleTrueFalse"
QUESTION_ANSWER = "QuestionAnswer"
CHARACTER_SPAN = "CharacterSpan"
CLASSIFICATION = "Classification"
UNDEFINED_QUESTION_TYPE = 0
MULTIPLE_CHOICE_ANSWER = "MultipleChoiceAnswer"
DIRECT_ANSWER = "DirectAnswer"
MULIPLE_LABELED_ANSWER = "MultipleLabelAnswer"

app = Flask(__name__)
#reader = WikiReader(1)

@app.route('/')
def index():
   return "KnowledgeBot API, I answer questions so ask away.\n"


class SolverServer:
    def __init__(self, solver):
        self.solver = solver
        self.answers_type = MULTIPLE_CHOICE_ANSWER
        if isinstance(self.solver, BidirectionalAttentionFlow):
            self.answers_multiple = DIRECT_ANSWER
        elif isinstance(self.solver, MultiClassificationModel):
            self.answers_type = MULIPLE_LABELED_ANSWER
        if K.backend() == "tensorflow":
            import tensorflow
            self.graph = tensorflow.get_default_graph()
            with self.graph.as_default():
                self.solver.load_model()
        else:
            self.solver.load_model()

    def read_instance_message(self, json_message):
        # pylint: disable=redefined-variable-type
        instance_type = json_message['type']
        if instance_type == TRUE_FALSE:
            text = json_message['text']
            instance = TextClassficationInstance(text, None, None)
        elif instance_type == QUESTION_ANSWER:
            question = json_message['text']
            options = json_message['answer_options']
            instance = QuestionAnswerInstance(question, options, None, None)
        elif instance_type == CHARACTER_SPAN:
            question = json_message['text']
            #passage = reader.get_background(question)
            passage = ""
            instance = CharacterSpanInstance(question, passage, None, None)
        elif instance_type == CLASSIFICATION:
            question = json_message['text']
            instance = MultipleLabelTextClassificationInstance(question, None)
        else:
            raise RuntimeError("Unrecognized instance type: " + instance_type)
        return instance

    def score_instance(self, instance):
        dataset = TextDataset([instance])
        return self.solver.score_text_dataset(dataset)

    def computeAnswer(self, request):
        instance = self.read_instance_message(request.json)
        try:
            if K.backend() == "tensorflow":
                with self.graph.as_default():
                    scores = self.score_instance(instance)
            else:
                scores = self.score_instance(instance)
        except:
            print("Instance was: " + str(instance))
            raise

        if self.answers_type == MULTIPLE_CHOICE_ANSWER:
            response = {
              'text': request.json['text'],
              'classes': scores.tolist(),
            }
        elif self.answers_type == MULIPLE_LABELED_ANSWER:
            predictions = scores.tolist()[0]
            labeled_scores = []
            for index, score in enumerate(predictions):
                label = self.solver.get_label_text(index)
                if (label != '@@PADDING@@') and (label != "@@UNKOWN@@"):
                    labeled_scores.append({'class_name':label, 'confidence': score})
            sorted_scores = sorted(labeled_scores, key=lambda k: k['confidence'], reverse=True)
            response = {
              'text': request.json['text'],
              'classes': sorted_scores[:10],
              'top_class': sorted_scores[0]['class_name']
            }
        else:
            begin_span_idx, end_span_idx = scores
            string_response = instance.passage_text[begin_span_idx:end_span_idx]
            response = {
              'type': request.json['text'],
              'classes': string_response
            }
        return jsonify(response)


@app.route('/kbot/api/v1.0/question', methods=['POST'])
def answerQuestion():
   if not request.json or not 'type' in request.json or not 'text' in request.json:
      abort(400)
   global mySolver
   return mySolver.computeAnswer(request)

@app.route('/kbot/api/v1.0/classify', methods=['POST'])
def classify():
   if not 'text' in request.json:
      abort(400)
   request.json['type'] = CLASSIFICATION
   global mySolver
   return mySolver.computeAnswer(request)

def serve(port: int, config_file: str):
    # read in the Typesafe-style config file
    solver_params = ConfigFactory.parse_file(config_file)
    params = Params(replace_none(solver_params))
    model_type = params.pop_choice( 'model_class', concrete_models.keys())
    solver_class = concrete_models[model_type]
    solver = solver_class(params)
    global mySolver
    mySolver = SolverServer(solver)
    # start the server on the specified port
    print("starting server")
    app.run()

def main():
    ensure_pythonhashseed_set()
    if len(sys.argv) != 3:
        print('USAGE: server.py [port] [config_file]')
        print('RECEIVED: ' + ' '.join(sys.argv))
        sys.exit(-1)
    port = int(sys.argv[1])
    config_file = sys.argv[2]
    serve(port, config_file)

if __name__ == '__main__':
   main()
