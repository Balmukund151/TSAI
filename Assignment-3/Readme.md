Final Validation accuracy for Base Network=82.56%

Best validation accuracy for my network=82.70%


******************************************************************************************************


def scheduler(epoch, lr):
  return round(0.01 * 1/(1 + 0.319 * epoch), 10)
  
  
# Define the model, size and Receptive Field are mentioned beside the layers of the model 
mymodel = Sequential()
mymodel.add(SeparableConv2D(32, 3, 3, depth_multiplier=2, input_shape=(32, 32, 3))) # size=30 Receptive Field=3
mymodel.add(BatchNormalization())
mymodel.add(Activation('relu'))
mymodel.add(SeparableConv2D(64, 3, 3, depth_multiplier=1)) # size=28 Receptive Field=5
mymodel.add(BatchNormalization())
mymodel.add(Activation('relu'))
mymodel.add(MaxPooling2D(pool_size=(2, 2))) #14 Receptive Field=6
mymodel.add(BatchNormalization())
mymodel.add(SeparableConv2D(64, 3, 3, depth_multiplier=2)) # size=12 Receptive Field=8
mymodel.add(BatchNormalization())
mymodel.add(Activation('relu'))
mymodel.add(Dropout(0.2))
mymodel.add(SeparableConv2D(128, 3, 3, depth_multiplier=2)) # size=10 Receptive Field=10
mymodel.add(BatchNormalization())
mymodel.add(Activation('relu'))
mymodel.add(Dropout(0.2))
mymodel.add(MaxPooling2D(pool_size=(2, 2))) # size=5 Receptive Field=11
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.2))
mymodel.add(SeparableConv2D(256, 3, 3, depth_multiplier=1)) # size=3 Receptive Field=13
mymodel.add(BatchNormalization())
mymodel.add(Activation('relu'))
mymodel.add(Dropout(0.2))
mymodel.add(SeparableConv2D(num_classes,3,3))
mymodel.add(Flatten())
mymodel.add(Activation('softmax'))
# Compile the model
mymodel.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
mymodel.summary()

mymodel_info = mymodel.fit_generator(datagen.flow(train_features, train_labels, batch_size = 256),
                                 samples_per_epoch = train_features.shape[0], nb_epoch = 50, 
                                 validation_data = (test_features, test_labels), 
                                 callbacks=[LearningRateScheduler(scheduler, verbose=1)],verbose=1)



*****************************************************************************************************************



/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:13: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  del sys.path[0]
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:13: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=(array([[[..., callbacks=[<keras.ca..., verbose=1, steps_per_epoch=195, epochs=50)`
  del sys.path[0]

Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.01.
195/195 [==============================] - 18s 91ms/step - loss: 1.2903 - acc: 0.5338 - val_loss: 2.6413 - val_acc: 0.3927
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0075815011.
195/195 [==============================] - 16s 81ms/step - loss: 0.8977 - acc: 0.6829 - val_loss: 1.4413 - val_acc: 0.5344
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0061050061.
195/195 [==============================] - 16s 81ms/step - loss: 0.7677 - acc: 0.7285 - val_loss: 1.3518 - val_acc: 0.5505
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.005109862.
195/195 [==============================] - 16s 81ms/step - loss: 0.6870 - acc: 0.7573 - val_loss: 0.8841 - val_acc: 0.6922
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0043936731.
195/195 [==============================] - 16s 81ms/step - loss: 0.6251 - acc: 0.7792 - val_loss: 0.8126 - val_acc: 0.7145
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0038535645.
195/195 [==============================] - 16s 81ms/step - loss: 0.5901 - acc: 0.7919 - val_loss: 0.6747 - val_acc: 0.7678
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.003431709.
195/195 [==============================] - 16s 81ms/step - loss: 0.5536 - acc: 0.8042 - val_loss: 0.7387 - val_acc: 0.7465
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0030931024.
195/195 [==============================] - 16s 81ms/step - loss: 0.5269 - acc: 0.8128 - val_loss: 0.6989 - val_acc: 0.7536
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0028153153.
195/195 [==============================] - 16s 81ms/step - loss: 0.5030 - acc: 0.8227 - val_loss: 0.7106 - val_acc: 0.7543
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0025833118.
195/195 [==============================] - 16s 81ms/step - loss: 0.4810 - acc: 0.8289 - val_loss: 0.6504 - val_acc: 0.7759
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0023866348.
195/195 [==============================] - 16s 81ms/step - loss: 0.4602 - acc: 0.8373 - val_loss: 0.6198 - val_acc: 0.7881
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0022177866.
195/195 [==============================] - 16s 81ms/step - loss: 0.4461 - acc: 0.8415 - val_loss: 0.5943 - val_acc: 0.7988
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.002071251.
195/195 [==============================] - 16s 81ms/step - loss: 0.4322 - acc: 0.8467 - val_loss: 0.6051 - val_acc: 0.7906
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0019428793.
195/195 [==============================] - 16s 82ms/step - loss: 0.4168 - acc: 0.8522 - val_loss: 0.6044 - val_acc: 0.7882
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0018294914.
195/195 [==============================] - 16s 81ms/step - loss: 0.4062 - acc: 0.8548 - val_loss: 0.6497 - val_acc: 0.7812
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0017286085.
195/195 [==============================] - 16s 81ms/step - loss: 0.3965 - acc: 0.8579 - val_loss: 0.6155 - val_acc: 0.7899
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.00163827.
195/195 [==============================] - 16s 81ms/step - loss: 0.3833 - acc: 0.8625 - val_loss: 0.5730 - val_acc: 0.8039
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0015569049.
195/195 [==============================] - 16s 81ms/step - loss: 0.3792 - acc: 0.8639 - val_loss: 0.6554 - val_acc: 0.7768
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0014832394.
195/195 [==============================] - 16s 81ms/step - loss: 0.3676 - acc: 0.8688 - val_loss: 0.5720 - val_acc: 0.8105
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.00141623.
195/195 [==============================] - 16s 80ms/step - loss: 0.3642 - acc: 0.8685 - val_loss: 0.5664 - val_acc: 0.8098
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0013550136.
195/195 [==============================] - 16s 81ms/step - loss: 0.3541 - acc: 0.8733 - val_loss: 0.5983 - val_acc: 0.7998
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.00129887.
195/195 [==============================] - 16s 80ms/step - loss: 0.3487 - acc: 0.8763 - val_loss: 0.5344 - val_acc: 0.8207
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0012471938.
195/195 [==============================] - 16s 81ms/step - loss: 0.3415 - acc: 0.8770 - val_loss: 0.5622 - val_acc: 0.8113
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0011994722.
195/195 [==============================] - 16s 81ms/step - loss: 0.3363 - acc: 0.8786 - val_loss: 0.5753 - val_acc: 0.8062
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.001155268.
195/195 [==============================] - 16s 81ms/step - loss: 0.3285 - acc: 0.8809 - val_loss: 0.5618 - val_acc: 0.8098
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0011142061.
195/195 [==============================] - 16s 81ms/step - loss: 0.3293 - acc: 0.8823 - val_loss: 0.5508 - val_acc: 0.8175
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.001075963.
195/195 [==============================] - 16s 81ms/step - loss: 0.3211 - acc: 0.8842 - val_loss: 0.5623 - val_acc: 0.8127
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.001040258.
195/195 [==============================] - 16s 81ms/step - loss: 0.3180 - acc: 0.8854 - val_loss: 0.5540 - val_acc: 0.8156
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0010068466.
195/195 [==============================] - 16s 81ms/step - loss: 0.3115 - acc: 0.8869 - val_loss: 0.5628 - val_acc: 0.8136
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0009755146.
195/195 [==============================] - 16s 81ms/step - loss: 0.3064 - acc: 0.8887 - val_loss: 0.5356 - val_acc: 0.8270
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0009460738.
195/195 [==============================] - 16s 81ms/step - loss: 0.3014 - acc: 0.8919 - val_loss: 0.5425 - val_acc: 0.8208
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.000918358.
195/195 [==============================] - 16s 81ms/step - loss: 0.3043 - acc: 0.8894 - val_loss: 0.5610 - val_acc: 0.8185
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0008922198.
195/195 [==============================] - 16s 81ms/step - loss: 0.3022 - acc: 0.8906 - val_loss: 0.5558 - val_acc: 0.8161
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0008675284.
195/195 [==============================] - 16s 81ms/step - loss: 0.2921 - acc: 0.8947 - val_loss: 0.5636 - val_acc: 0.8140
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0008441668.
195/195 [==============================] - 16s 81ms/step - loss: 0.2890 - acc: 0.8951 - val_loss: 0.5463 - val_acc: 0.8190
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0008220304.
195/195 [==============================] - 16s 81ms/step - loss: 0.2874 - acc: 0.8966 - val_loss: 0.6183 - val_acc: 0.7987
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0008010253.
195/195 [==============================] - 16s 81ms/step - loss: 0.2897 - acc: 0.8941 - val_loss: 0.5522 - val_acc: 0.8194
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0007810669.
195/195 [==============================] - 16s 81ms/step - loss: 0.2835 - acc: 0.8971 - val_loss: 0.5828 - val_acc: 0.8098
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.000762079.
195/195 [==============================] - 16s 81ms/step - loss: 0.2791 - acc: 0.8982 - val_loss: 0.5899 - val_acc: 0.8090
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0007439923.
195/195 [==============================] - 16s 81ms/step - loss: 0.2759 - acc: 0.8995 - val_loss: 0.5540 - val_acc: 0.8202
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0007267442.
195/195 [==============================] - 16s 81ms/step - loss: 0.2743 - acc: 0.9019 - val_loss: 0.5506 - val_acc: 0.8195
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0007102777.
195/195 [==============================] - 16s 81ms/step - loss: 0.2767 - acc: 0.8987 - val_loss: 0.5371 - val_acc: 0.8218
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0006945409.
195/195 [==============================] - 16s 81ms/step - loss: 0.2715 - acc: 0.8996 - val_loss: 0.5820 - val_acc: 0.8120
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0006794863.
195/195 [==============================] - 16s 81ms/step - loss: 0.2704 - acc: 0.9013 - val_loss: 0.5607 - val_acc: 0.8166
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0006650705.
195/195 [==============================] - 16s 81ms/step - loss: 0.2667 - acc: 0.9032 - val_loss: 0.5612 - val_acc: 0.8168
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0006512537.
195/195 [==============================] - 16s 82ms/step - loss: 0.2644 - acc: 0.9031 - val_loss: 0.5539 - val_acc: 0.8213
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0006379992.
195/195 [==============================] - 16s 81ms/step - loss: 0.2640 - acc: 0.9044 - val_loss: 0.5390 - val_acc: 0.8228
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0006252736.
195/195 [==============================] - 16s 81ms/step - loss: 0.2587 - acc: 0.9058 - val_loss: 0.5822 - val_acc: 0.8116
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0006130456.
195/195 [==============================] - 16s 81ms/step - loss: 0.2631 - acc: 0.9042 - val_loss: 0.5605 - val_acc: 0.8185
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0006012868.
195/195 [==============================] - 16s 81ms/step - loss: 0.2579 - acc: 0.9067 - val_loss: 0.5592 - val_acc: 0.8188
Model took 792.65 seconds to train

Accuracy on test data is: 81.88

								 