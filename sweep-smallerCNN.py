from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten
from keras.layers import Dropout, Dense
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import Sequential
from keras import preprocessing
import wandb
from wandb.keras import WandbCallback

sweep_config = {
  'method': 'bayes', 
  'metric': {
      'name': 'val_loss',
      'goal': 'minimize'
  },
  'parameters': {
      'batch_size': {
          'values': [2, 4, 6, 8, 16, 32]
      },
      'dropout_c': {
          'values': [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
      },
      'dropout_ga': {
          'values': [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
      },
      'num_features_c1': {
          'values': [8, 16, 32, 64, 128]
      },
      'num_features_c2': {
          'values': [8, 16, 32, 64, 128]
      },
      'num_features_c3': {
          'values': [8, 16, 32, 64, 128]
      },
      'learning_rate':{
          'values': [0.00001, 0.00005, 0.0001, 0.00025, 0.0005, 0.00075, 0.001]
      },
      'epochs':{
          'values': [10, 20, 30, 40, 50, 60, 70, 80]
      }
  }
}

def sweep_train():
    # file to save model weights and structure to
    model_file = 'smallCNN-sweep.h5'

    # Specify the hyperparameters to be tuned along with
    # an initial value
    config_defaults={
        "batch_size": 4, # number of images processed during training before weights update
        "epochs": 5, # number of times the model passes through the entire training dataset
        "dropout_c": 0.3, # dropout from convolutional layers
        "dropout_ga": 0.1, # dropout from global averaging layer
        "num_features_c1": 64, # ""
        "num_features_c2": 64, # sizes of feature maps in respective convolutional blocks
        "num_features_c3": 128, # ""
        "learning_rate": 0.0003
    }

    # Initialize wandb with a sample project name
    
    wandb.init(config=config_defaults)  # this gets over-written in the Sweep
    # Specify the other hyperparameters to the configuration, if any
    wandb.config.architecture_name = "CNN"

    (img_height, img_width) = (640, 480)
    input_shape = (img_width, img_height, 3)


    # data is split into three categories:
    # training for tuning the network
    # validation to check performance during training
    # testing to check performance after training
    train_dir = "images/train"
    val_dir = "images/validate"
    test_dir = "images/test"

    # process data from the sub-directories into categories (diarrhea, farts, pee, and poop)
    datagen = preprocessing.image.ImageDataGenerator(
        rescale=1./255)

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=wandb.config.batch_size,
        class_mode='categorical')

    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=wandb.config.batch_size,
        class_mode='categorical')

    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=wandb.config.batch_size,
        class_mode='categorical')

    # model consists of 4 convolutional blocks, each following a similar structure:
    # 1. convolutional layer
    # 2. max pooling layer
    # 3. dropout
    # the activations of the final convolutional block are pooled together in the global average pooling layer.
    # the network then reduces to 4 final neurons, providing probability outputs for each category.
    model = Sequential()

    model.add(Conv2D(wandb.config.num_features_c1, (3, 3), input_shape=input_shape, activation="relu", padding="valid"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(wandb.config.dropout_c))
    
    model.add(Conv2D(wandb.config.num_features_c2, (3, 3), activation="relu", padding="valid"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(wandb.config.dropout_c))
    
    model.add(Conv2D(wandb.config.num_features_c3, (3, 3), activation="relu", padding="valid"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(wandb.config.dropout_c))

    model.add(GlobalAveragePooling2D())
    model.add(Dropout(wandb.config.dropout_ga))
    model.add(Dense(4, activation="softmax"))

    model.summary()
    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(learning_rate=wandb.config.learning_rate),
                metrics=['accuracy'])

    # the model during training with the lowest validation loss is saved as the final model
    modelCheckpoint = ModelCheckpoint(model_file, monitor='val_loss', mode='min', save_best_only=True)
    history = model.fit(train_generator, validation_data=val_generator, epochs=wandb.config.epochs, batch_size=wandb.config.batch_size, 
                callbacks=[WandbCallback(), modelCheckpoint])


# Initialize the sweep, count is how many sweep iterations to do
sweep_id = wandb.sweep(sweep_config, project="smallCNN-sweep-FT")
wandb.agent(sweep_id, function=sweep_train, count=100)