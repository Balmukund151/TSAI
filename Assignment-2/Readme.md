model.evaluate gives 99.44% accuracy
Total params: 14,052

************************************************************************************************************************************
Logs for 20 epochs:-

Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.005.
60000/60000 [==============================] - 21s 350us/step - loss: 0.1676 - acc: 0.9475 - val_loss: 0.1179 - val_acc: 0.9661
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0037907506.
60000/60000 [==============================] - 15s 254us/step - loss: 0.0654 - acc: 0.9802 - val_loss: 0.0712 - val_acc: 0.9780
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0030525031.
60000/60000 [==============================] - 15s 255us/step - loss: 0.0537 - acc: 0.9834 - val_loss: 0.0417 - val_acc: 0.9869
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.002554931.
60000/60000 [==============================] - 15s 254us/step - loss: 0.0455 - acc: 0.9862 - val_loss: 0.0367 - val_acc: 0.9872
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0021968366.
60000/60000 [==============================] - 15s 255us/step - loss: 0.0411 - acc: 0.9872 - val_loss: 0.0276 - val_acc: 0.9916
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0019267823.
60000/60000 [==============================] - 15s 254us/step - loss: 0.0368 - acc: 0.9885 - val_loss: 0.0268 - val_acc: 0.9915
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0017158545.
60000/60000 [==============================] - 15s 253us/step - loss: 0.0331 - acc: 0.9898 - val_loss: 0.0281 - val_acc: 0.9912
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0015465512.
60000/60000 [==============================] - 15s 252us/step - loss: 0.0317 - acc: 0.9899 - val_loss: 0.0330 - val_acc: 0.9891
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0014076577.
60000/60000 [==============================] - 15s 253us/step - loss: 0.0294 - acc: 0.9911 - val_loss: 0.0223 - val_acc: 0.9928
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0012916559.
60000/60000 [==============================] - 15s 253us/step - loss: 0.0267 - acc: 0.9918 - val_loss: 0.0200 - val_acc: 0.9931
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0011933174.
60000/60000 [==============================] - 15s 254us/step - loss: 0.0259 - acc: 0.9920 - val_loss: 0.0215 - val_acc: 0.9934
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.0011088933.
60000/60000 [==============================] - 15s 257us/step - loss: 0.0236 - acc: 0.9923 - val_loss: 0.0234 - val_acc: 0.9923
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0010356255.
60000/60000 [==============================] - 16s 262us/step - loss: 0.0227 - acc: 0.9926 - val_loss: 0.0230 - val_acc: 0.9930
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0009714397.
60000/60000 [==============================] - 16s 259us/step - loss: 0.0215 - acc: 0.9930 - val_loss: 0.0197 - val_acc: 0.9938
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0009147457.
60000/60000 [==============================] - 15s 258us/step - loss: 0.0211 - acc: 0.9930 - val_loss: 0.0219 - val_acc: 0.9938
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0008643042.
60000/60000 [==============================] - 15s 257us/step - loss: 0.0197 - acc: 0.9936 - val_loss: 0.0213 - val_acc: 0.9935
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000819135.
60000/60000 [==============================] - 15s 257us/step - loss: 0.0192 - acc: 0.9939 - val_loss: 0.0208 - val_acc: 0.9937
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0007784524.
60000/60000 [==============================] - 16s 259us/step - loss: 0.0189 - acc: 0.9936 - val_loss: 0.0189 - val_acc: 0.9937
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0007416197.
60000/60000 [==============================] - 16s 263us/step - loss: 0.0185 - acc: 0.9940 - val_loss: 0.0207 - val_acc: 0.9941
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000708115.
60000/60000 [==============================] - 15s 257us/step - loss: 0.0167 - acc: 0.9947 - val_loss: 0.0205 - val_acc: 0.9944

<keras.callbacks.History at 0x7faf4377d908>



******************************************************************************************************************************

Strategy
1. Made batch size=64. this improved accuracy though took more training time per epoch.
2. set learning rate in Adam optimiser= 0.005 [This is to start with higher learning rate at the beginning and decrease the learning rate using scheduler to suit as the accuracy reaches optimum level as number of epochs are fixed to 20]
3. reduced the number of filters in order to decrease the Total Params.
4. I tried not to alter anything else in the original model like adding pooling or more conv layers. i have just tried to optimize the existing solution to reduce number of parameters and improve the accuracy and see the effect of various parameter changes.