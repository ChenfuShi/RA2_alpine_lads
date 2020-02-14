from tensorflow.keras.layers import Dense

from dataset.joint_dataset import feet_joint_dataset
from utils import top_2_categorical_accuracy

def train_feet_erosion_model(config, pretrained_base_model):
    dataset = feet_joint_dataset(config)
    feet_joint_erosion_dataset, val_dataset = dataset.create_feet_joints_dataset(True, val_split = True)

    # Add erosion outcomes to pretrained model
    pretrained_base_model.add(Dense(5, activation = 'softmax', name = 'main_output'))

    pretrained_base_model.compile(loss = 'categorical_crossentropy', metrics=["categorical_accuracy", top_2_categorical_accuracy], optimizer='adam')

    history = pretrained_base_model.fit(
        feet_joint_erosion_dataset, epochs = 25, steps_per_epoch = 200, validation_data = val_dataset, validation_steps = 5, class_weight = dataset.class_weights
    )

    return pretrained_base_model
