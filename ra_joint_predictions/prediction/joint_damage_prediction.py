import numpy as np

import model.joint_damage_model as joint_damage_model

class joint_damage_predictor():
    def __init__(self, model_parameters):
        self.model_file = model_parameters['model']
        self.no_outcomes = model_parameters['no_outcomes']
        self.no_classes = model_parameters['no_classes']
        self.is_wrist = model_parameters.get('is_wrist', False)
        self.model_type = model_parameters['model_type']

        self.joint_damage_prediction_model = joint_damage_model.load_joint_damage_model(self.model_file, self.no_classes, model_type = self.model_type)

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

def _tansform_classification_prediction(joint_damage_predictor, prediction):
    # Calculate the output as softmax weighted sum of possible outcomes
    prediction = np.sum(prediction * np.arange(joint_damage_predictor.no_classes))

    return prediction

def _transform_regression_prediction(joint_damage_predictor, prediction):
    # Make sure the regressed scores are actual possible values
    prediction = np.max([prediction, 0])
    prediction = np.min([prediction, joint_damage_predictor.no_classes - 1])

    return prediction