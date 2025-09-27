# uses CNN
import tensorflow as tf
import tensorflow_datasets as tfds

(ds_train, ds_val, ds_test), ds_info = tfds.load(
    'mnist',
    split = ['train[:80%]', 'train[80%:90%]', 'train[90%:]'], #80-10-10 split for train, val, test
    shuffle_files = True,
    as_supervised = True, # gives tuple (image, label) instead of dict
    with_info = True
)

def normalize_img(image, label):
    return tf.cast(image, tf.float32)/255., label

# training pipeline
#tf.data.Dataset.map: normalizes images from type tf.uint9 to tf.float32
ds_train = ds_train.map(normalize_img, num_parallel_calls = tf.data.AUTOTUNE)

#tf.data.Dataset.cache: cache before shuffling for better performance
ds_train = ds_train.cache()

#tf.data.Dataset.shuffle: shuffles with shuffle buffer = full dataset size for true randomness
ds_train = ds_train.shuffle(ds_info.splits['train[:80%]'].num_examples)

#tf.data.Dataset.batch: get unique batches at each epoch by batching elements after shuffling
ds_train = ds_train.batch(128)

#tf.data.Dataset.prefetch: end pipeline by prefetching for good performance
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


ds_val = ds_val.map(normalize_img, num_parallel_calls = tf.data.AUTOTUNE)
ds_val = ds_val.batch(128)
ds_val = ds_val.cache()
ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

# evaluation pipeline
#do caching after bactching because batches can be the same between epochs
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)



# creating and training the model
model = tf.keras.models.Sequential([
    # a convolutional layer with 32 filters that look at a 3x3 patch of pixesl at a time
    # activation = 'relu' introduces non-linearity so the model can learn complex shapes
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(28,28,1)),
    # another convolutional layer this time with 64 filtersn so the network learns more complex features by combining the first layer's simple ones
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    # downsamples the feature maps (halves width and height) to make the network faster and more translation invariant
    tf.keras.layers.MaxPooling2D(),
    # randomly sets 25% of neurons to 0 during training to force network to not rely on any one filter too much (reduces overfitting)
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(input_shape = (28,28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation = 'softmax') 
])
model.compile(
    optimizer = tf.keras.optimizers.Adam(0.001),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs = 15,
    validation_data = ds_val,
)

# model evaluation
test_loss, test_accuracy = model.evaluate(ds_test)
print()
print("Test Loss:", test_loss) #0.0401
print("Test Accuracy:", test_accuracy) # 0.9903

train_loss, train_accuracy = model.evaluate(ds_train)
print()
print("Train Loss:", train_loss) #0.00247
print("Train Accuracy:", train_accuracy) # 0.99927

val_loss, val_accuracy = model.evaluate(ds_val)
print()
print("Validation Loss:", val_loss) # 0.0351
print("Validation Accuracy:", val_accuracy) # 0.991
