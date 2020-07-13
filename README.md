weights used:

#### original model for landmarks 
Face Pretraining: FACE_original_retrained_model_100

hands landmark prediciton:
HANDS_train_original_model_1000 
resources/hands_landmarks_original_epoch_1000_predictor_1.h5

feet landmark prediction:
FEET_train_original_model_1000
resources/feet_landmarks_original_epoch_1000_predictor_1.h5


#### resnet model for landmarks 
seems to not be as good as the original model
faces pretraining:
FACE_pretrain_resnet_model_250

hands landmark prediciton:
HANDS_train_resnet_model_2500
  
feet landmark prediction:
HANDS_train_resnet_model_2500







#### joint classfication model "complex"
NIH pretraining:
NIH_new_pretrain_model_250



#### joint classfication model resnet
NIH pretraining:
NIH_resnet_pretrain_model_250






Dependencies

tensorflow 2.0.0

tensorflow-gpu 2.0.0

tensorflow-addons 0.6.0

Pillow

Pandas

Matplotlib

opencv
