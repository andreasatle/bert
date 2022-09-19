"""
Workflow for running model optimization.
"""
from dataset import Dataset
from model import Model
from optimizer import Optimizer


def run(epochs=5, init_lr=3e-5, model_name="small_bert/bert_en_uncased_L-4_H-512_A-8"):
    """
    Executes the workflow for model optimization.
    """
    dataset = Dataset()
    print(dataset)
    model = Model(model_name)
    opt = Optimizer(dataset, epochs, init_lr)
    model.compile(opt)
    _history = model.fit(dataset, opt)
    model.save("bert.h5")
    # model.save_to_json("bert-json-dir")
