from data import Data
from model import Model
from optimizer import Optimizer

def run():
    data = Data()
    model = Model()
    opt = Optimizer(data)
    model.compile(opt)
    history = model.fit(data, opt)
    model.save('model')
