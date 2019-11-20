model.evaluate gives 99.46% accuracy
Total params: 14,060

************************************************************************************************************************************
Logs for 20 epochs:-

Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 41s 691us/step - loss: 0.4672 - acc: 0.8558 - val_loss: 0.0696 - val_acc: 0.9828
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 39s 647us/step - loss: 0.2609 - acc: 0.9124 - val_loss: 0.0516 - val_acc: 0.9855
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 39s 646us/step - loss: 0.2142 - acc: 0.9270 - val_loss: 0.0466 - val_acc: 0.9859
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 39s 650us/step - loss: 0.1907 - acc: 0.9335 - val_loss: 0.0344 - val_acc: 0.9908
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 39s 650us/step - loss: 0.1756 - acc: 0.9380 - val_loss: 0.0295 - val_acc: 0.9922
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 39s 651us/step - loss: 0.1650 - acc: 0.9400 - val_loss: 0.0288 - val_acc: 0.9922
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 39s 651us/step - loss: 0.1562 - acc: 0.9438 - val_loss: 0.0252 - val_acc: 0.9935
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 39s 644us/step - loss: 0.1495 - acc: 0.9446 - val_loss: 0.0242 - val_acc: 0.9922
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 38s 629us/step - loss: 0.1452 - acc: 0.9444 - val_loss: 0.0247 - val_acc: 0.9925
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 38s 629us/step - loss: 0.1430 - acc: 0.9455 - val_loss: 0.0260 - val_acc: 0.9926
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 38s 627us/step - loss: 0.1358 - acc: 0.9485 - val_loss: 0.0229 - val_acc: 0.9935
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 37s 624us/step - loss: 0.1355 - acc: 0.9475 - val_loss: 0.0249 - val_acc: 0.9932
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 38s 628us/step - loss: 0.1340 - acc: 0.9485 - val_loss: 0.0220 - val_acc: 0.9943
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 38s 630us/step - loss: 0.1286 - acc: 0.9497 - val_loss: 0.0209 - val_acc: 0.9942
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 38s 630us/step - loss: 0.1270 - acc: 0.9498 - val_loss: 0.0230 - val_acc: 0.9935
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 38s 638us/step - loss: 0.1251 - acc: 0.9493 - val_loss: 0.0207 - val_acc: 0.9939
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 38s 631us/step - loss: 0.1249 - acc: 0.9502 - val_loss: 0.0232 - val_acc: 0.9938
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 38s 627us/step - loss: 0.1247 - acc: 0.9511 - val_loss: 0.0202 - val_acc: 0.9943
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 38s 631us/step - loss: 0.1225 - acc: 0.9517 - val_loss: 0.0224 - val_acc: 0.9937
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 38s 630us/step - loss: 0.1169 - acc: 0.9530 - val_loss: 0.0207 - val_acc: 0.9946

<keras.callbacks.History at 0x7fe52410e588>

******************************************************************************************************************************

Strategy
1. Made batch size=32. this improved accuracy though took more training time per epoch.
2. set learning rate in Adam optimiser= 0.009 [This is to start with higher learning rate at the beginning and decrease the learning rate using scheduler to suit as the accuracy reaches optimum level as number of epochs are fixed to 20]
3. reduced the number of filters in order to decrease the Total Params.
4. I tried not to alter anything else in the original model like adding pooling or more conv layers. i have just tried to optimize the existing solution to reduce number of parameters and improve the accuracy and see the effect of various parameter changes.