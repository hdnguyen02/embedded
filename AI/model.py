import tensorflow as tf

def create_neural_network(image_width,image_height, channel):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape = (image_width, image_height, channel )),
        tf.keras.layers.Rescaling(scale=1./255),
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.5),    
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),    
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def create_model(image_width,image_height, channel):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape = (image_width, image_height, channel )),
        tf.keras.layers.Rescaling(scale=1./255),
        tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.5),    
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def create_1D_neural_network(lenght, class_num):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape = lenght),
        tf.keras.layers.Rescaling(scale=1./255),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(0.5),    
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.5),    
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    return model

def create_1D_model(lenght):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape = lenght),
        tf.keras.layers.Rescaling(scale=1./255),
        tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.5),    
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model