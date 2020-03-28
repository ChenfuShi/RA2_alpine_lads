import numpy as np

import model.joint_damage_model as joint_damage_model

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