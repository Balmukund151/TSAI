model.evaluate gives 99.52% accuracy
Total params: 14,754

************************************************************************************************************************************
Logs for 20 epochs:-

Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 11s 182us/step - loss: 0.5385 - acc: 0.8513 - val_loss: 0.1070 - val_acc: 0.9790
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 6s 99us/step - loss: 0.2590 - acc: 0.9236 - val_loss: 0.0694 - val_acc: 0.9841
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 6s 98us/step - loss: 0.2047 - acc: 0.9394 - val_loss: 0.0470 - val_acc: 0.9896
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 6s 98us/step - loss: 0.1778 - acc: 0.9444 - val_loss: 0.0412 - val_acc: 0.9896
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 6s 97us/step - loss: 0.1569 - acc: 0.9487 - val_loss: 0.0367 - val_acc: 0.9915
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 6s 98us/step - loss: 0.1441 - acc: 0.9504 - val_loss: 0.0304 - val_acc: 0.9929
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 6s 97us/step - loss: 0.1341 - acc: 0.9523 - val_loss: 0.0289 - val_acc: 0.9925
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 6s 97us/step - loss: 0.1283 - acc: 0.9536 - val_loss: 0.0306 - val_acc: 0.9914
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 6s 97us/step - loss: 0.1224 - acc: 0.9535 - val_loss: 0.0235 - val_acc: 0.9931
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 6s 96us/step - loss: 0.1164 - acc: 0.9546 - val_loss: 0.0261 - val_acc: 0.9939
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 6s 96us/step - loss: 0.1152 - acc: 0.9539 - val_loss: 0.0258 - val_acc: 0.9925
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 6s 97us/step - loss: 0.1114 - acc: 0.9558 - val_loss: 0.0213 - val_acc: 0.9940
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 6s 95us/step - loss: 0.1068 - acc: 0.9554 - val_loss: 0.0217 - val_acc: 0.9941
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 6s 97us/step - loss: 0.1078 - acc: 0.9555 - val_loss: 0.0211 - val_acc: 0.9942
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 6s 98us/step - loss: 0.1029 - acc: 0.9572 - val_loss: 0.0221 - val_acc: 0.9940
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 6s 96us/step - loss: 0.1027 - acc: 0.9560 - val_loss: 0.0188 - val_acc: 0.9945
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 6s 99us/step - loss: 0.1039 - acc: 0.9548 - val_loss: 0.0209 - val_acc: 0.9938
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 6s 99us/step - loss: 0.1009 - acc: 0.9557 - val_loss: 0.0185 - val_acc: 0.9950
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 6s 100us/step - loss: 0.0975 - acc: 0.9574 - val_loss: 0.0197 - val_acc: 0.9950
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 6s 96us/step - loss: 0.1002 - acc: 0.9559 - val_loss: 0.0198 - val_acc: 0.9952


******************************************************************************************************************************

Strategy
1. Removed the 1*1 layer so that full information of previous channels can be passed on.
2. reduced 2nd layer from 32 filter to 16 filters to reduce the number of parameters. Though it reduced the channels from 32 to 16 but since i removed the 10 filters of 1*1 (mentioned in 1st point) , hence all the channels information from this conv layer 2 was passed on to next stages.
4. I tried not to alter anything else in the original model like adding pooling or more conv layers. i have just tried to optimize the existing solution to reduce number of parameters and improve the accuracy and see the effect of various parameter changes.