import logging
import numpy as np
import tensorflow as tf

import model.joint_damage_model as joint_damage_model
import dataset.ops.image_ops as img_ops

from dataset.ops.dataset_ops import _augment_and_clip_image

joint_pred_augments = [
    {
        'augment': img_ops.random_flip,
        'p': 1
    },
    {
        'augment': img_ops.random_brightness_and_contrast
    },
    {
        'augment': img_ops.random_crop
    },
    {
        'augment': img_ops.random_rotation
    }
]

wrist_pred_augments = [
    {
        'augment': img_ops.random_brightness_and_contrast
    },
    {
        'augment': img_ops.random_crop,
        'params': {'min_scale': 0.9}
    },
    {
        'augment': img_ops.random_rotation,
        'params': {'max_rot': 10}
    }
]

def _default_transformation(prediction):
    return prediction

def _robust_mean(scores, rounding_cutoff = 0):
    N, D = scores.shape
    
    mean_scores = np.zeros(D)
    labels = ['N'] * D
    
    for d in range(D):
        values = scores[:, d]
        
        values = np.sort(values)
        
        start_idx = N // 10
        end_idx = N - start_idx
        
        sub_values = values[start_idx:end_idx]
        mean_score = np.mean(sub_values)
        
        # Augments introduce noise so round down to 0 if it's at that cutoff
        if mean_score <= rounding_cutoff:
            mean_score = 0.0
            labels[d] = 'R'
        
        mean_scores[d] = mean_score
    
    return mean_scores, labels

class predictor():
    def __init__(self, model_file, no_outcomes = 1, is_wrist = False, prediction_transformer = _default_transformation, model_index = 0):
        self.model_file = model_file
        self.no_outcomes = no_outcomes
        self.is_wrist = is_wrist
        self.prediction_transformer = prediction_transformer
        self.model_index = model_index
        
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
        model_file = self.model_file[self.model_index]
        
        self.model = tf.keras.models.load_model(f'../resources/{model_file}', compile = False)

class joint_damage_predictor(predictor):
    def __init__(self, model_parameters, model_index = 0):
        super().__init__(model_parameters['model'], no_outcomes = model_parameters['no_outcomes'], is_wrist = model_parameters.get('is_wrist', False), model_index = model_index)

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
            if np.isnan(prediction):
                prediction = 0
            elif np.isinf(prediction):
                prediction = self.no_classes - 1
            else:
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
    def __init__(self, model_parameters, model_index = 0):
        super().__init__(model_parameters['damage_type_model'], model_index = model_index)

        self.model_parameters = model_parameters
        self.prediction_transformer = self._create_sig_prediction_transformer()

    def _create_sig_prediction_transformer(self):
        def _sig_prediction_transformer(prediction):
            if np.isnan(prediction):
                prediction = 0.0
            elif np.isinf(prediction):
                prediction = 1.0

            return prediction

        return _sig_prediction_transformer

class ensembled_filter_predictor():
    def __init__(self, model_parameters, aggregator = _robust_mean, rounding_cutoff = 0, no_augments = 50):
        self.model_parameters = model_parameters

        self.no_outcomes = model_parameters['no_outcomes']
        self.is_wrist = model_parameters.get('is_wrist', False)

        self.no_pred_models = model_parameters.get('no_pred_models', 0)
        self.no_filter_models = model_parameters.get('no_filter_models', 0)
        
        self.filter_cutoff = self.model_parameters.get('damage_type_cutoff', 0.2)
        # Value to return if the filter_prediction exceeds the cutoff
        self.default_value = self.model_parameters.get('default_value', 0.0)
        
        self.aggregator = _robust_mean
        self.rounding_cutoff = rounding_cutoff
        self.no_augments = no_augments
        
        self.augments = []

        self.n_filtered_images = 0
        self.n_processed_images = 0

        self._init_model()

    def _init_model(self):
        def _get_predictor(model_index, pred_type):
            predictor = pred_type(self.model_parameters, model_index = model_index)
            
            return predictor
        
        self.damage_predictors = list(map(lambda no_model: _get_predictor(no_model, joint_damage_predictor), range(self.no_pred_models)))
        self.filter_predictors = list(map(lambda no_model: _get_predictor(no_model, joint_damage_type_predictor), range(self.no_filter_models)))

        if self.is_wrist:
            self.augments = img_ops.create_augments(wrist_pred_augments)
        else:
            self.augments = img_ops.create_augments(joint_pred_augments)
    
    def predict_joint_damage(self, img):
        self.n_processed_images += self.no_outcomes

        no_preds = self.no_augments + 1

        dmg_preds = np.zeros((len(self.damage_predictors * no_preds), self.no_outcomes))
        filter_preds = np.zeros((len(self.filter_predictors * no_preds), self.no_outcomes))
        
        dmg_idx = 0
        filter_idx = 0

        for n in range(self.no_augments + 1):
            if n > 0:
                pred_img, _ = _augment_and_clip_image(img, [], augments = self.augments)
            else:
                pred_img = img

            for damage_predictor in self.damage_predictors:
                dmg_preds[dmg_idx] = damage_predictor.predict_joint_damage(pred_img)
                dmg_idx += 1
            
            for filter_predictor in self.filter_predictors:
                filter_preds[filter_idx] = filter_predictor.predict_joint_damage(pred_img)
                filter_idx += 1

        dmg_preds, dmg_pred_labels = self.aggregator(dmg_preds, self.rounding_cutoff)
              
        if self.no_filter_models > 0:
            filter_preds, _ = self.aggregator(filter_preds)
        
            filtered_idx = np.where(filter_preds <= self.filter_cutoff)[0]
        
            dmg_preds[filtered_idx] = self.default_value
            for idx in filtered_idx:
                dmg_pred_labels[idx] = 'F'

            self.n_filtered_images += filtered_idx.size
        
        return dmg_preds, dmg_pred_labels