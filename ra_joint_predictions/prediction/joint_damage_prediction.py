import logging
import numpy as np
import tensorflow as tf

import model.joint_damage_model as joint_damage_model
import dataset.ops.image_ops as img_ops

from dataset.ops.dataset_ops import _augment_and_clip_image

pred_augments = [
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

class augmented_predictor():
    def __init__(self, base_predictor, no_augments = 50, aggregator = _robust_mean, rounding_cutoff = 0):
        self.base_predictor = base_predictor
        self.no_augments = no_augments
        self.aggregator = aggregator
        self.rounding_cutoff = rounding_cutoff
        
        augments = pred_augments
        
        if base_predictor.is_wrist:
            augments = augments[1:]
        
        self.augments = img_ops.create_augments(augments)

    def predict_joint_damage(self, img):
        preds = np.zeros((self.no_augments + 1, self.base_predictor.no_outcomes))
        
        for n in range(self.no_augments):
            aug_img, _ = _augment_and_clip_image(img, [], augments = self.augments)
            preds[n] = self.base_predictor.predict_joint_damage(aug_img)

        preds[self.no_augments, :] = self.base_predictor.predict_joint_damage(img)
        
        self.scores = preds
        
        if self.aggregator is not None:
            preds = self.aggregator(preds, self.rounding_cutoff)
        
        return preds
    
class ensemble_predictor():
    def __init__(self, model_parameters, pred_type, no_models, aggregator = _robust_mean, rounding_cutoff = 0, no_augments = 50):
        self.model_parameters = model_parameters
        
        self.pred_type = pred_type
        self.no_models = no_models
        
        self.aggregator = _robust_mean
        self.rounding_cutoff = rounding_cutoff
        self.no_augments = no_augments
        
        self._init_model()
        
    def _init_model(self):
        def _get_predictor(model_index):
            predictor = self.pred_type(self.model_parameters, model_index = model_index)
            predictor = augmented_predictor(predictor, aggregator = None, no_augments = self.no_augments)
            
            return predictor
        
        self.predictors = list(map(_get_predictor, range(self.no_models)))
        
    def predict_joint_damage(self, img):
        preds = []
        
        for predictor in self.predictors:
            pred = predictor.predict_joint_damage(img)
            
            preds.extend(pred)
            
        preds = np.array(preds)
            
        return self.aggregator(preds, self.rounding_cutoff)
    
class filtered_joint_damage_predictor():
    def __init__(self, model_parameters, filter_predictor, follow_up_predictor):
        self.model_parameters = model_parameters
        self.cutoff = self.model_parameters.get('damage_type_cutoff', 0.2)
        # Value to return if the filter_prediction exceeds the cutoff
        self.default_value = self.model_parameters.get('default_value', 0.0)

        self.filter_predictor = filter_predictor
        self.follow_up_predictor = follow_up_predictor
        
        self.n_processed_images = 0
        self.n_filtered_images = 0

    def predict_joint_damage(self, img):
        self.n_processed_images += 1
        
        sig_pred = self.filter_predictor.predict_joint_damage(img)[0]
        
        # If the probability of it not being 0 is > cutoff, pass it on to the next predictor
        if sig_pred > self.cutoff:
            return self.follow_up_predictor.predict_joint_damage(img)
        else:
            self.n_filtered_images += 1
            
            return [0.0], ['F']
