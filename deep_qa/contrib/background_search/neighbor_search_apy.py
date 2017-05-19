#!flask/bin/python
from flask import Flask, jsonify, request
import logging
import itertools
import os
import sys

# These have to be before we do any import from keras.  It would be nice to be able to pass in a
# value for this, but that makes argument passing a whole lot more complicated.  If/when we change
# how arguments work (e.g., loading a file), then we can think about setting this as a parameter.
import random
import numpy
random.seed(13370)
numpy.random.seed(1337)  # pylint: disable=no-member
# pylint: disable=wrong-import-position

import pyhocon

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from deep_qa.common.checks import ensure_pythonhashseed_set
from deep_qa.common.params import Params, replace_none
from deep_qa.common import util
from deep_qa.contrib.background_search.vector_based_retrieval import VectorBasedRetrieval

app = Flask(__name__)

def get_neighbors(retrieval: VectorBasedRetrieval,
                                 question: str,
                                 num_neighbors: int):
    nearest_neighbors = retrieval.get_nearest_neighbors(question, num_neighbors)
    return nearest_neighbors


@app.route('/kbot/api/v1.0/semantic_search', methods=['POST'])
def search():
   if not 'text' in request.json:
      abort(400)
   global retrieval
   neighbors = get_neighbors(retrieval, request.json['text'], 10)
   # JSONIFY the results
   response = {
     'question': request.json['text'],
     'neighbors': neighbors
   }
   return jsonify(response)

def serve(port: int, param_file: str):
    # read in the Typesafe-style config file
    params = pyhocon.ConfigFactory.parse_file(param_file)
    params = Params(replace_none(params))
    retrieval_params = params.pop('retrieval')
    corpus_file = params.pop('corpus', None)
    num_neighbors = params.pop('num_neighbors', 10)

    global retrieval
    retrieval = VectorBasedRetrieval(retrieval_params)
    if corpus_file is not None:
        retrieval.read_background(corpus_file)
        retrieval.fit()
        retrieval.save_model()
    else:
        retrieval.load_model()

    # start the server on the specified port
    print("starting server")
    app.run(host= '0.0.0.0')

def main():
    ensure_pythonhashseed_set()
    if len(sys.argv) != 3:
        print('USAGE: neighbor_search_api.py [port] [config_file]')
        print('RECEIVED: ' + ' '.join(sys.argv))
        sys.exit(-1)
    port = int(sys.argv[1])
    config_file = sys.argv[2]
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)
    serve(port, config_file)

if __name__ == '__main__':
   main()
