"""
This module contains the BERT-model.
"""
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as _
import tensorflowjs as tfjs
import bert_maps as bert


class Model:
    """
    A class that contains the BERT-model.
    """

    def __init__(self, model_name):
        """
        Initialize the Model class.
        """

        print("model_name:", model_name)

        # === Choose a BERT model
        tfhub_handle_encoder, tfhub_handle_preprocess = _choose_a_bert_model(model_name)
        print("Encoder:", tfhub_handle_encoder)
        print("Pre-process:", tfhub_handle_preprocess)

        # === Preprocess model
        bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

        # === Try the preprocessed model
        text_test = ["this is such an amazing movie!"]
        text_preprocessed = _try_preprocessed_model(bert_preprocess_model, text_test)

        # === Load model from hub
        bert_model = hub.KerasLayer(tfhub_handle_encoder)

        # === Evaluate the model
        _evaluate_preprocessed_model(
            bert_model, text_preprocessed, tfhub_handle_encoder
        )

        # === Build classifier model
        self.model = _build_classifier_model(
            tfhub_handle_preprocess, tfhub_handle_encoder
        )

    def __str__(self):
        """
        Output string for the Model class.
        """

        return self.__class__.__name__

    def compile(self, opt):
        """
        Compile the BERT-model.
        """
        self.model.compile(optimizer=opt.optimizer, loss=opt.loss, metrics=opt.metrics)

    def fit(self, data, opt):
        """
        Fit the BERT-model.
        """
        history = self.model.fit(
            x=data.train_ds, validation_data=data.val_ds, epochs=opt.epochs
        )
        return history

    def save(self, model_name):
        """
        Save the BERT-model.
        """
        self.model.save(model_name, include_optimizer=False)

    def load(self, model_name):
        """
        Load the BERT-model.
        """
        self.model = tf.saved_model.load(model_name)

    # This hasn't been validated to work yet.
    def save_to_json(self, model_name):
        """
        Save the model to the json-format.
        """
        tfjs.converters.save_keras_model(self.model, model_name)


# === Choose a BERT model
def _choose_a_bert_model(bert_model_name):

    tfhub_handle_encoder = bert.map_name_to_handle[bert_model_name]
    tfhub_handle_preprocess = bert.map_model_to_preprocess[bert_model_name]

    print(f"BERT model selected           : {tfhub_handle_encoder}")
    print(f"Preprocess model auto-selected: {tfhub_handle_preprocess}")

    return tfhub_handle_encoder, tfhub_handle_preprocess


# === Try the preprocessed model
def _try_preprocessed_model(bert_preprocess_model, text_test):
    text_preprocessed = bert_preprocess_model(text_test)

    print(f"Keys       : {list(text_preprocessed.keys())}")
    print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
    print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
    print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
    print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

    return text_preprocessed


# === Evaluate the model
def _evaluate_preprocessed_model(bert_model, text_preprocessed, tfhub_handle_encoder):
    bert_results = bert_model(text_preprocessed)

    print(f"Loaded BERT: {tfhub_handle_encoder}")
    print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
    print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
    print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
    print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')


# === Build classifier model
def _build_classifier_model(tfhub_handle_preprocess, tfhub_handle_encoder):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name="preprocessing")
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name="BERT_encoder")
    outputs = encoder(encoder_inputs)
    net = outputs["pooled_output"]
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name="classifier")(net)

    return tf.keras.Model(text_input, net)
