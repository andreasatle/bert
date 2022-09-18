from data import Data
from model import Model
from optimizer import Optimizer


def run(epochs=5, init_lr=3e-5, model_name="small_bert/bert_en_uncased_L-4_H-512_A-8"):
    data = Data()
    model = Model(model_name)
    model.summary()
    opt = Optimizer(data, epochs, init_lr)
    model.compile(opt)
    history = model.fit(data, opt)
    model.save("bert.h5")
    model.save_to_json("bert-json-dir")
