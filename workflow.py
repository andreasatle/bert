"""
Workflow for running model optimization.
"""
from data import Data
from model import Model
from optimizer import Optimizer


def run(epochs=5, init_lr=3e-5, model_name="small_bert/bert_en_uncased_L-4_H-512_A-8"):
    """
    Executes the workflow for model optimization.
    """
    data = Data()
    print(data)
    model = Model(model_name)
    opt = Optimizer(data, epochs, init_lr)
    model.compile(opt)
    _history = model.fit(data, opt)
    model.save("bert.h5")
    # model.save_to_json("bert-json-dir")
