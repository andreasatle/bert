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