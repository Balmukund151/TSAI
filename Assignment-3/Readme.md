Final Validation accuracy for Base Network=82.56%

Best validation accuracy for MY network=83.37% (on 44th Epoch)


******************************************************************************************************


from keras.layers import SeparableConv2D
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(0.01 * 1/(1 + 0.309 * epoch), 10)
  
  
# Define the model, size and Receptive Field are mentioned beside the layers of the model 
mymodel = Sequential()
mymodel.add(SeparableConv2D(32, 3, 3, depth_multiplier=2, input_shape=(32, 32, 3))) # size=30 Receptive Field=3
mymodel.add(BatchNormalization())
mymodel.add(Activation('relu'))
mymodel.add(SeparableConv2D(64, 3, 3, depth_multiplier=1)) # size=28 Receptive Field=5
mymodel.add(BatchNormalization())
mymodel.add(Activation('relu'))
mymodel.add(Dropout(0.1))
mymodel.add(SeparableConv2D(64, (3, 3), strides=2,dilation_rate=2,depth_multiplier=2)) # size=12 Receptive Field=13
mymodel.add(BatchNormalization())
mymodel.add(Activation('relu'))
mymodel.add(Dropout(0.2))
mymodel.add(SeparableConv2D(128, 3, 3, depth_multiplier=1)) # size=10 Receptive Field=15
mymodel.add(BatchNormalization())
mymodel.add(Activation('relu'))
mymodel.add(Dropout(0.2))
mymodel.add(SeparableConv2D(128, (3, 3),depth_multiplier=1)) # size=8 Receptive Field=17
mymodel.add(BatchNormalization())
mymodel.add(Activation('relu'))
mymodel.add(Dropout(0.2))
mymodel.add(SeparableConv2D(128, (3, 3),depth_multiplier=1)) # size=6 Receptive Field=19
mymodel.add(BatchNormalization())
mymodel.add(Activation('relu'))
mymodel.add(Dropout(0.2))
mymodel.add(SeparableConv2D(num_classes,6,6))
mymodel.add(Flatten())
mymodel.add(Activation('softmax'))
# Compile the model
mymodel.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
mymodel.summary()

datagen = ImageDataGenerator(zoom_range=0.0,
                             horizontal_flip=True)


# train the model
start = time.time()
# Train the model
mymodel_info = mymodel.fit_generator(datagen.flow(train_features, train_labels, batch_size = 128),
                                 samples_per_epoch = train_features.shape[0], nb_epoch = 50,
                                 validation_data = (test_features, test_labels), 
                                 callbacks=[LearningRateScheduler(scheduler, verbose=1)],verbose=1)



*****************************************************************************************************************




/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:13: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  del sys.path[0]
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:13: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=(array([[[..., callbacks=[<keras.ca..., verbose=1, steps_per_epoch=390, epochs=50)`
  del sys.path[0]

Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.01.
390/390 [==============================] - 34s 88ms/step - loss: 1.4587 - acc: 0.4709 - val_loss: 2.6211 - val_acc: 0.3642
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0076394194.
390/390 [==============================] - 20s 52ms/step - loss: 1.0895 - acc: 0.6155 - val_loss: 1.1319 - val_acc: 0.6031
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0061804697.
390/390 [==============================] - 20s 52ms/step - loss: 0.9277 - acc: 0.6740 - val_loss: 0.9085 - val_acc: 0.6798
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0051894136.
390/390 [==============================] - 21s 53ms/step - loss: 0.8299 - acc: 0.7087 - val_loss: 1.1584 - val_acc: 0.6343
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0044722719.
390/390 [==============================] - 20s 52ms/step - loss: 0.7610 - acc: 0.7312 - val_loss: 0.7842 - val_acc: 0.7299
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0039292731.
390/390 [==============================] - 20s 52ms/step - loss: 0.7091 - acc: 0.7508 - val_loss: 0.8271 - val_acc: 0.7176
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0035038542.
390/390 [==============================] - 20s 52ms/step - loss: 0.6649 - acc: 0.7668 - val_loss: 0.7612 - val_acc: 0.7315
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0031615555.
390/390 [==============================] - 20s 53ms/step - loss: 0.6358 - acc: 0.7766 - val_loss: 0.6979 - val_acc: 0.7633
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0028801843.
390/390 [==============================] - 20s 52ms/step - loss: 0.6116 - acc: 0.7860 - val_loss: 0.6869 - val_acc: 0.7661
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.002644803.
390/390 [==============================] - 20s 52ms/step - loss: 0.5867 - acc: 0.7942 - val_loss: 0.6579 - val_acc: 0.7746
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0024449878.
390/390 [==============================] - 20s 52ms/step - loss: 0.5711 - acc: 0.8017 - val_loss: 0.6955 - val_acc: 0.7648
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0022732439.
390/390 [==============================] - 20s 52ms/step - loss: 0.5567 - acc: 0.8060 - val_loss: 0.6200 - val_acc: 0.7917
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0021240442.
390/390 [==============================] - 20s 52ms/step - loss: 0.5358 - acc: 0.8122 - val_loss: 0.6064 - val_acc: 0.7967
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.001993223.
390/390 [==============================] - 20s 52ms/step - loss: 0.5250 - acc: 0.8163 - val_loss: 0.5778 - val_acc: 0.8075
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0018775817.
390/390 [==============================] - 20s 52ms/step - loss: 0.5177 - acc: 0.8178 - val_loss: 0.5777 - val_acc: 0.8052
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0017746229.
390/390 [==============================] - 20s 52ms/step - loss: 0.5013 - acc: 0.8227 - val_loss: 0.5550 - val_acc: 0.8122
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.0016823688.
390/390 [==============================] - 20s 52ms/step - loss: 0.4951 - acc: 0.8249 - val_loss: 0.5918 - val_acc: 0.8057
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0015992324.
390/390 [==============================] - 20s 52ms/step - loss: 0.4873 - acc: 0.8294 - val_loss: 0.5610 - val_acc: 0.8143
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0015239256.
390/390 [==============================] - 20s 52ms/step - loss: 0.4756 - acc: 0.8303 - val_loss: 0.6203 - val_acc: 0.7973
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.0014553922.
390/390 [==============================] - 20s 52ms/step - loss: 0.4729 - acc: 0.8323 - val_loss: 0.5797 - val_acc: 0.8046
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0013927577.
390/390 [==============================] - 20s 52ms/step - loss: 0.4683 - acc: 0.8338 - val_loss: 0.5766 - val_acc: 0.8090
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.0013352918.
390/390 [==============================] - 20s 52ms/step - loss: 0.4626 - acc: 0.8368 - val_loss: 0.5492 - val_acc: 0.8170
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0012823801.
390/390 [==============================] - 20s 52ms/step - loss: 0.4478 - acc: 0.8408 - val_loss: 0.5768 - val_acc: 0.8099
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0012335019.
390/390 [==============================] - 20s 53ms/step - loss: 0.4477 - acc: 0.8407 - val_loss: 0.5283 - val_acc: 0.8253
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0011882129.
390/390 [==============================] - 20s 52ms/step - loss: 0.4404 - acc: 0.8447 - val_loss: 0.5339 - val_acc: 0.8208
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0011461318.
390/390 [==============================] - 20s 52ms/step - loss: 0.4388 - acc: 0.8439 - val_loss: 0.5339 - val_acc: 0.8244
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0011069294.
390/390 [==============================] - 20s 52ms/step - loss: 0.4351 - acc: 0.8451 - val_loss: 0.5678 - val_acc: 0.8128
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.00107032.
390/390 [==============================] - 20s 52ms/step - loss: 0.4321 - acc: 0.8486 - val_loss: 0.5475 - val_acc: 0.8184
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0010360547.
390/390 [==============================] - 20s 52ms/step - loss: 0.4230 - acc: 0.8500 - val_loss: 0.5366 - val_acc: 0.8258
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0010039153.
390/390 [==============================] - 20s 52ms/step - loss: 0.4219 - acc: 0.8506 - val_loss: 0.5352 - val_acc: 0.8254
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0009737098.
390/390 [==============================] - 20s 52ms/step - loss: 0.4143 - acc: 0.8517 - val_loss: 0.5306 - val_acc: 0.8255
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0009452689.
390/390 [==============================] - 20s 52ms/step - loss: 0.4136 - acc: 0.8538 - val_loss: 0.5138 - val_acc: 0.8300
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0009184423.
390/390 [==============================] - 21s 53ms/step - loss: 0.4103 - acc: 0.8533 - val_loss: 0.5352 - val_acc: 0.8230
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0008930964.
390/390 [==============================] - 20s 52ms/step - loss: 0.4064 - acc: 0.8548 - val_loss: 0.5422 - val_acc: 0.8243
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0008691118.
390/390 [==============================] - 20s 53ms/step - loss: 0.4032 - acc: 0.8569 - val_loss: 0.5472 - val_acc: 0.8231
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0008463817.
390/390 [==============================] - 21s 53ms/step - loss: 0.4047 - acc: 0.8579 - val_loss: 0.5224 - val_acc: 0.8315
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0008248103.
390/390 [==============================] - 21s 53ms/step - loss: 0.4014 - acc: 0.8568 - val_loss: 0.5292 - val_acc: 0.8278
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0008043111.
390/390 [==============================] - 20s 52ms/step - loss: 0.3963 - acc: 0.8610 - val_loss: 0.5430 - val_acc: 0.8250
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0007848062.
390/390 [==============================] - 20s 52ms/step - loss: 0.3969 - acc: 0.8606 - val_loss: 0.5265 - val_acc: 0.8325
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0007662248.
390/390 [==============================] - 21s 53ms/step - loss: 0.3892 - acc: 0.8614 - val_loss: 0.5272 - val_acc: 0.8307
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.000748503.
390/390 [==============================] - 20s 52ms/step - loss: 0.3868 - acc: 0.8628 - val_loss: 0.5180 - val_acc: 0.8306
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0007315824.
390/390 [==============================] - 20s 52ms/step - loss: 0.3877 - acc: 0.8618 - val_loss: 0.5242 - val_acc: 0.8283
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0007154099.
390/390 [==============================] - 20s 52ms/step - loss: 0.3845 - acc: 0.8636 - val_loss: 0.5409 - val_acc: 0.8245
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.000699937.
390/390 [==============================] - 20s 52ms/step - loss: 0.3859 - acc: 0.8616 - val_loss: 0.5219 - val_acc: 0.8337
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0006851192.
390/390 [==============================] - 20s 52ms/step - loss: 0.3789 - acc: 0.8635 - val_loss: 0.5204 - val_acc: 0.8332
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0006709158.
390/390 [==============================] - 20s 52ms/step - loss: 0.3795 - acc: 0.8656 - val_loss: 0.5239 - val_acc: 0.8315
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0006572893.
390/390 [==============================] - 21s 53ms/step - loss: 0.3759 - acc: 0.8654 - val_loss: 0.5250 - val_acc: 0.8296
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0006442054.
390/390 [==============================] - 21s 53ms/step - loss: 0.3817 - acc: 0.8652 - val_loss: 0.5303 - val_acc: 0.8321
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0006316321.
390/390 [==============================] - 20s 52ms/step - loss: 0.3732 - acc: 0.8682 - val_loss: 0.5194 - val_acc: 0.8310
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0006195403.
390/390 [==============================] - 21s 53ms/step - loss: 0.3738 - acc: 0.8681 - val_loss: 0.5203 - val_acc: 0.8304
Model took 1035.30 seconds to train

Accuracy on test data is: 83.04



								 