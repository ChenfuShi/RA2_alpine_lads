## A two-stage model to classify joint damage in radiographs

This repository contains the source code developed for our submission to the RA2 DREAM Challenge: Automated Scoring of Radiographic Joint Damage: https://www.synapse.org/#!Synapse:syn20545111/wiki/.

Various trained models are available from the ./trained_models directory. Refer to ./ra_joint_predictions/dream/dream_model_parameters_collection.json to see which models were used in the final predictions. 

Some models used exceed the size restrictions of github, please reach out to use if you require these models for your research. Alternatively, all models used in the final submission are available from our challenge page: https://www.synapse.org/#!Synapse:syn21610007/wiki/604496 

### Dependencies
tensorflow 2.0.0

tensorflow-gpu 2.0.0

tensorflow-addons 0.6.0

Pillow
Pandas

Matplotlib

opencv

keras-adamw (Available here: https://github.com/OverLordGoldDragon/keras-adamw - this repository contains a copy (./ra_joint_predictions/keras-adamw) of the keras-adamw repo, to fix some issues, that have since been fixed in the original code)
