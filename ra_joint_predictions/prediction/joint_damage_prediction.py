import logging
import numpy as np
import tensorflow as tf

import model.joint_damage_model as joint_damage_model

from dataset.ops.dataset_ops import _augment_and_clip_image

def _default_transformation(prediction):
    return prediction

def _robust_mean(scores):
    N = scores.shape[1]
        
    mean_score = np.zeros(N)
    for n in range(N):
        n_scores = scores[:, n]
        n_scores = n_scores[~np.isnan(n_scores)]
            
        if n_scores.size != 0:
            max_score = np.percentile(n_scores, 90)
            min_score = np.percentile(n_scores, 10)

            filtered_min = np.extract(n_scores >= min_score, n_scores)
            filtered_max = np.extract(filtered_min <= max_score, filtered_min)
            mean_score[n] = np.mean(filtered_max)
        else:
            logging.warn('All preds nan - output 0!')
                
            mean_score[n] = 0
        
    return mean_score

class predictor():
    def __init__(self, model_file, no_outcomes = 1, is_wrist = False, prediction_transformer = _default_transformation):
        self.model_file = model_file
        self.no_outcomes = no_outcomes
        self.is_wrist = False
        self.prediction_transformer = prediction_transformer

        self._init_model()

    def predict_joint_damage(self, img):
        img = tf.expand_dims(img, 0)

        predicted_joint_damage = np.zeros(self.no_outcomes)
        y_preds = self.model.predict(img)

        for n in range(self.no_outcomes):
            y_pred = y_preds[n][0]

            if self.is_wrist:
                y_pred = y_pred[0]

            y_pred = self.prediction_transformer(y_pred)

            predicted_joint_damage[n] = y_pred

        return predicted_joint_damage

    def _init_model(self):
        self.model = tf.keras.models.load_model(self.model_file, compile = False)

class joint_damage_predictor(predictor):
    def __init__(self, model_parameters):
        super().__init__(model_parameters['model'], no_outcomes = model_parameters['no_outcomes'], is_wrist = model_parameters.get('is_wrist', False))

        self.model_parameters = model_parameters
        self.no_classes = model_parameters['no_classes']
        self.model_type = model_parameters['model_type']

        self._init_prediction_transformer()

    def _init_prediction_transformer(self):
        if self.model_type == joint_damage_model.MODEL_TYPE_REGRESSION:
            self.prediction_transformer = self._create_regression_prediction_transformer()
        else:
            self.prediction_transformer = self._create_classifciation_prediction_transformer()
    
    def _create_regression_prediction_transformer(self):
        def _regression_prediction_transformer(prediction):
            # Make sure the regressed scores are actual possible values
            prediction = np.max([prediction, 0])
            prediction = np.min([prediction, self.no_classes - 1])

            return prediction

        return _regression_prediction_transformer

    def _create_classifciation_prediction_transformer(self):
        def _classifciation_prediction_transformer(prediction):
            # Calculate the output as softmax weighted sum of possible outcomes
            prediction = np.sum(prediction * np.arange(self.no_classes))

            return prediction

        return _classifciation_prediction_transformer

class joint_damage_type_predictor(predictor):
    def __init__(self, model_parameters):
        super().__init__(model_parameters['damage_type_model'])

        self.model_parameters = model_parameters

class augmented_predictor():
    def __init__(self, base_predictor, no_augments = 50, aggregator = _robust_mean):
        self.base_predictor = base_predictor
        self.no_augments = no_augments
        self.aggregator = aggregator

    def predict_joint_damage(self, img):
        preds = np.zeros((self.no_augments + 1, self.base_predictor.no_outcomes))
        
        for n in range(self.no_augments):
            aug_img, _ = _augment_and_clip_image(img, [])
            preds[n] = self.base_predictor.predict_joint_damage(aug_img)

        preds[self.no_augments, :] = self.base_predictor.predict_joint_damage(img)
        
        return self.aggregator(preds)

class augmented_joint_damage_predictor(joint_damage_predictor):
    def __init__(self, model_parameters, no_augments = 50, aggregator = _robust_mean):
        super().__init__(model_parameters)
        
        # Number of times to augment the image
        self.no_augments = no_augments

        # Function that aggregates the augmented predictions into a single prediction
        self.aggregator = aggregator
        
    def predict_joint_damage(self, img):
        preds = np.zeros((self.no_augments + 1, self.no_outcomes))
        
        for n in range(self.no_augments):
            aug_img, _ = _augment_and_clip_image(img, [])

            aug_img_pred = super().predict_joint_damage(aug_img)
            
            preds[n] = aug_img_pred

        preds[self.no_augments, :] = super().predict_joint_damage(img)
        
        return self.aggregator(preds)
    
class filtered_joint_damage_predictor():
    def __init__(self, model_parameters, filter_predictor, follow_up_predictor):
        self.model_parameters = model_parameters
        self.cutoff = self.model_parameters.get('damage_type_cutoff', 0.2)
        # Value to return if the filter_prediction exceeds the cutoff
        self.default_value = self.model_parameters.get('default_value', 0.0)

        self.filter_predictor = filter_predictor
        self.follow_up_predictor = follow_up_predictor

    def predict_joint_damage(self, img):
        sig_pred = self.filter_predictor.predict_joint_damage(img)[0]
        
        # If the probability of it not being 0 is > cutoff, pass it on to the next predictor
        if sig_pred > self.cutoff:
            return self.follow_up_predictor.predict_joint_damage(img)
        else:
            return [0.0]