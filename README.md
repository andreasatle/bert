# Refactor of tensorflow example "Classify text with BERT"

[Classify text with BERT](https://www.tensorflow.org/text/tutorials/classify_text_with_bert)

The main workflow is:
```
data = Data()
model = Model()
opt = Optimizer(data)
model.compile(opt)
history = model.fit(data, opt)
model.save('model')
```

After a while I got the same error on both Mac and windows(ubuntu). We have to set the directory for the cache in tensorflow_hub.
```
export TFHUB_CACHE_DIR=$HOME/.cache/tfhub_modules
```

Also, it is necessary to import the module ```tensorflow_text```,
```
import tensorflow_text as _
```
even though it's not explicitly being used in the module in ```model.py```.

To avoid writing ```__pycache__``` every time you load a python-module, try:
```
import sys
sys.dont_write_bytecode = True
```
before importing any modules. I created a ```runme.py``` for this.