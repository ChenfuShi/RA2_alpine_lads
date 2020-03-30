import logging
import numpy as np
import tensorflow as tf

import model.joint_damage_model as joint_damage_model

from dataset.ops.dataset_ops import _augment_and_clip_image

class joint_damage_predictor():
    def __init__(self, model_parameters):
        self.model_parameters = model_parameters
        
        self.model_file = model_parameters['model']
        self.no_outcomes = model_parameters['no_outcomes']
        self.no_classes = model_parameters['no_classes']
        self.is_wrist = model_parameters.get('is_wrist', False)
        self.model_type = model_parameters['model_type']

        self.joint_damage_prediction_model = joint_damage_model.load_joint_damage_model(self.model_file)

        if self.model_type == joint_damage_model.MODEL_TYPE_REGRESSION:
            self.prediction_transformation = _transform_regression_prediction
        else:
            self.prediction_transformation = _tansform_classification_prediction

    def predict_joint_damage(self, img):
        predicted_joint_damage = np.zeros(self.no_outcomes)

        y_preds = self.joint_damage_prediction_model.predict(img)
        
        for n in range(self.no_outcomes):
            y_pred = y_preds[n][0]

            if self.is_wrist:
                y_pred = y_pred[0]

            y_pred = self.prediction_transformation(self, y_pred)

            predicted_joint_damage[n] = y_pred

        return predicted_joint_damage

class augmented_joint_damage_predictor(joint_damage_predictor):
    def __init__(self, model_parameters, no_augments = 50):
        super().__init__(model_parameters)
        
        self.no_augments = no_augments
        
    def predict_joint_damage(self, img):
        preds = np.zeros((self.no_augments + 1, self.no_outcomes))
        
        for n in range(self.no_augments):
            aug_img, _ = _augment_and_clip_image(img, [])
            aug_img = tf.expand_dims(aug_img, 0)
            
            aug_img_pred = super().predict_joint_damage(aug_img)
            
            preds[n] = aug_img_pred
            
        img = tf.expand_dims(img, 0)
        preds[self.no_augments, :] = super().predict_joint_damage(img)
        
        pred = self._robust_mean(preds)
        
        return pred
    
    def _robust_mean(self, scores):
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
    
class filtered_joint_damage_predictor():
    def __init__(self, joint_damage_predictor):
        self.model_parameters = joint_damage_predictor.model_parameters
        
        self.joint_damage_predictor = joint_damage_predictor

        self.cutoff = self.model_parameters.get('damage_type_cutoff', 0.3)
        self.damage_type_model_file = self.model_parameters['damage_type_model']

        self.joint_damage_type_prediction_model = joint_damage_model.load_joint_damage_model(self.damage_type_model_file)

    def predict_joint_damage(self, img):
        sig_pred = self.joint_damage_type_prediction_model.predict(img)[0][0]
        
        # If the probability of it not being 0 is > cutoff, pass it on to the next predictor
        if sig_pred > self.cutoff:
            return self.joint_damage_predictor.predict_joint_damage(img)
        else:
            return [0.0]

def _tansform_classification_prediction(joint_damage_predictor, prediction):
    # Calculate the output as softmax weighted sum of possible outcomes
    prediction = np.sum(prediction * np.arange(joint_damage_predictor.no_classes))

    return prediction

def _transform_regression_prediction(joint_damage_predictor, prediction):
    # Make sure the regressed scores are actual possible values
    prediction = np.max([prediction, 0])
    prediction = np.min([prediction, joint_damage_predictor.no_classes - 1])

    return prediction