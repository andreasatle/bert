import tensorflow as tf
from official.nlp import optimization  # to create AdamW optimizer


class Optimizer:
    def __init__(self, data, epochs=5, init_lr=3e-5):
        self.epochs = epochs
        self.init_lr = init_lr

        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.metrics = tf.metrics.BinaryAccuracy()

        steps_per_epoch = tf.data.experimental.cardinality(data.train_ds).numpy()
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(0.1 * num_train_steps)

        self.optimizer = optimization.create_optimizer(
            init_lr=init_lr,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            optimizer_type="adamw",
        )
