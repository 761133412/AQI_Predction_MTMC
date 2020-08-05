from keras.models import Input, Model, Sequential                     #### TCN
from keras.layers import Dense, Activation, Conv1D, LSTM, Dropout, Reshape, Bidirectional, Flatten, Add, Concatenate, MaxPool1D, LeakyReLU
from keras.callbacks import Callback
from Support.support_tcn import TCN
# from ranger import Ranger

from Support.support_nested_lstm import NestedLSTM      ##### NLSTM
from Support.support_OL import ONLSTM


import time


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.totaltime = time.time()

    def on_train_end(self, logs={}):
        self.totaltime = time.time() - self.totaltime

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def buildTCN_LSTM(timestep):

    batch_size, timesteps, input_dim = None, timestep, 1
    i = Input(batch_shape=(batch_size, timesteps, input_dim))
    x = TCN(return_sequences=False)(i)  # The TCN layers are here.
    #x = Dense(32)(x)

    x = Reshape((-1, 1))(x)
    x = LSTM(output_dim=100, return_sequences=True)(x)
    print(x.shape)
    x = LSTM(100)(x)
    x = Dropout(0.2)(x)

    o = Dense(1)(x)
    o = Activation('linear')(o)
    #output_layer = x
    model = Model(inputs=[i], outputs=[o])
    # model.summary()
    model.compile(optimizer='rmsprop', loss='mse',)

    return model


def build_LSTM(time_step):
    model = Sequential()            # layers = [1, ahead_, 100, 1]
    model.add(LSTM(
            input_shape=(time_step, 1),
            output_dim=time_step,
            # units=layers[2],
            return_sequences=True))
        #model.add(Dropout(0.2))
    model.add(LSTM(
            100,
            return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(
            output_dim=1))
    model.add(Activation("linear"))

    model.summary()
    model.compile(loss="mse", optimizer="rmsprop")
    return model


# def buildNLSTM(timestep):
#
#     batch_size, timesteps, input_dim = None, timestep, 1
#     i = Input(batch_shape=(batch_size, timesteps, input_dim))
#
#     # x = Reshape((-1, 1, 1))(i)
#     x = NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1)(i)
#     x = Dense(16, activation='linear')(x)
#
#     # x = Dense(64)(i)
#     # x = Bidirectional(NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1))(x)
#     print(x.shape)
#     # x = LSTM(100)(x)
#
#     # o = Dense(1)(x)
#     # o = Activation('linear')(o)
#     o = Dense(1, activation="linear")(x)
#     # output_layer = x
#     # model = Sequential()
#     model = Model(inputs=[i], outputs=[o])
#
#     model.compile(optimizer='rmsprop', loss='mse', )
#     model.summary()
#
#     return model


def buildNLSTM(timestep):

    batch_size, timesteps, input_dim = None, timestep, 1
    i = Input(batch_shape=(batch_size, timesteps, input_dim))

    # x = Reshape((-1, 1, 1))(i)
    x = NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1)(i)
    x = Dense(16, activation='linear')(x)

    # x = Dense(64)(i)
    # x = Bidirectional(NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1))(x)
    print(x.shape)
    # x = LSTM(100)(x)

    # o = Dense(1)(x)
    # o = Activation('linear')(o)
    o = Dense(1, activation="linear")(x)
    # output_layer = x
    # model = Sequential()
    model = Model(inputs=[i], outputs=[o])
    # optimizer = Ranger(model.parameters(), **kwargs)
    model.compile(optimizer='rmsprop', loss='mse', )
    model.summary()

    return model


def buildBLSTM(timestep):

    batch_size, timesteps, input_dim = None, timestep, 1
    i = Input(batch_shape=(batch_size, timesteps, input_dim))

    # x = Reshape((-1, 1, 1))(i)
    # x = NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1)(i)

    # x = Dense(64)(i)
    # x = Bidirectional(LSTM(output_dim=64, return_sequences=True))(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(i)
    x = Dropout(0.2)(x)
    print(x.shape)
    # x = LSTM(100)(x)
    x = Flatten()(x)
    o = Dense(1)(x)
    o = Activation('linear')(o)
    # output_layer = x
    # model = Sequential()
    model = Model(inputs=[i], outputs=[o])

    model.compile(optimizer='rmsprop', loss='mse', )
    model.summary()

    return model


def buildSLSTM(timestep):
    batch_size, timesteps, input_dim = None, timestep, 1
    i = Input(batch_shape=(batch_size, timesteps, input_dim))

    x = Reshape((-1, 1))(i)
    x = LSTM(output_dim=128, return_sequences=True)(x)
    x = LSTM(output_dim=64, return_sequences=True)(x)
    x = LSTM(output_dim=64, return_sequences=True)(x)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)

    o = Dense(1)(x)
    o = Activation('linear')(o)
    # output_layer = x
    model = Model(inputs=[i], outputs=[o])

    model.compile(optimizer='rmsprop', loss='mse', )
    model.summary()
    return model


def buildOLSTM(timestep):

    batch_size, timesteps, input_dim = None, timestep, 1

    i = Input(batch_shape=(batch_size, timesteps, input_dim))
    # i = Input(batch_shape=(batch_size, timesteps))

    # x = Reshape((timesteps))(i)

    x = Reshape((-1, 1))(i)

    # x = ONLSTM(64, timesteps, return_sequences=True, dropconnect=0.25)(x)

    for ix in range(3):
        onlstm = ONLSTM(64, timesteps, return_sequences=True, dropconnect=0.10)
        x = onlstm(x)

    # x = Dense(64)(x)
    # x = Bidirectional(NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1))(x)
    print(x.shape)
    # x = LSTM(100)(x)

    x = Dense(1)(x)

    print(x.shape)
    x = Flatten()(x)

    x = Dense(1)(x)
    o = Activation('linear')(x)
    # output_layer = x
    # model = Sequential()
    model = Model(inputs=[i], outputs=[o])

    model.compile(optimizer='rmsprop', loss='mse', )
    model.summary()

    return model


def buildNLSTM_MultiV_v1(timestep):

    batch_size, timesteps, input_dim = None, timestep, 1

    input_A3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x0 = NestedLSTM(48, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_A3)
    x0 = Dense(16, activation='linear')(x0)
    model0 = Model(inputs=input_A3, outputs=x0)

    input_D1 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x1 = NestedLSTM(48, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_D1)
    x1 = Dense(16, activation='linear')(x1)
    model1 = Model(inputs=input_D1, outputs=x1)

    input_D2 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x2 = NestedLSTM(48, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_D2)
    x2 = Dense(16, activation='linear')(x2)
    model2 = Model(inputs=input_D2, outputs=x2)

    input_D3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x3 = NestedLSTM(48, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_D3)
    x3 = Dense(16, activation='linear')(x3)
    model3 = Model(inputs=input_D3, outputs=x3)

    # o = Add()([model0.output, model1.output, model2.output, model3.output])

    combined = Concatenate(axis=1)([model0.output, model1.output, model2.output, model3.output])
    o = Dense(16, activation="linear")(combined)

    o = Dense(1, activation="linear")(o)

    model = Model(inputs=[model0.input, model1.input, model2.input, model3.input], outputs=o)

    model.compile(optimizer='rmsprop', loss='mse')
    model.summary()

    return model


def buildNLSTM_MultiV_v2(timestep):
    batch_size, timesteps, input_dim = None, timestep, 1

    input_0 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x0 = NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_0)
    x0 = Dense(16, activation='linear')(x0)
    model0 = Model(inputs=input_0, outputs=x0)

    input_1 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x1 = NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_1)
    x1 = Dense(16, activation='linear')(x1)
    model1 = Model(inputs=input_1, outputs=x1)

    input_2 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x2 = NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_2)
    x2 = Dense(16, activation='linear')(x2)
    model2 = Model(inputs=input_2, outputs=x2)

    input_3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x3 = NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_3)
    x3 = Dense(16, activation='linear')(x3)
    model3 = Model(inputs=input_3, outputs=x3)

    input_4 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x4 = NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_4)
    x4 = Dense(16, activation='linear')(x4)
    model4 = Model(inputs=input_4, outputs=x4)

    input_5 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x5 = NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_5)
    x5 = Dense(16, activation='linear')(x5)
    model5 = Model(inputs=input_5, outputs=x5)

    combined = Concatenate(axis=1)([model0.output,
                                    model1.output,
                                    model2.output,
                                    model3.output,
                                    model4.output,
                                    model5.output])

    output0 = Dense(1, activation='linear')(combined)

    output1 = Dense(1, activation='linear')(combined)

    output2 = Dense(1, activation='linear')(combined)

    output3 = Dense(1, activation='linear')(combined)

    output4 = Dense(1, activation='linear')(combined)

    output5 = Dense(1, activation='linear')(combined)

    model = Model(inputs=[model0.input,
                          model1.input,
                          model2.input,
                          model3.input,
                          model4.input,
                          model5.input],
                  outputs=[output0,
                           output1,
                           output2,
                           output3,
                           output4,
                           output5]
                  )

    model.compile(optimizer='rmsprop', loss='mse', )
    model.summary()

    return model


# 定义多通道特征组合模型
def build_MC_CNN_LSTM(timestep):
    # 定义输入
    batch_size, timesteps, input_dim = None, timestep, 1
    i = Input(batch_shape=(batch_size, timesteps, input_dim))

    # ########################################
    # cnn层
    cnn_out1 = Conv1D(filters=8, kernel_size=2, strides=1, padding='same')(i)
    cnn_out1 = Conv1D(filters=8, kernel_size=2, strides=1, padding='same')(cnn_out1)
    cnn_out1 = LeakyReLU()(cnn_out1)

    cnn_out2 = Conv1D(filters=8, kernel_size=3, strides=1, padding='same')(i)
    cnn_out2 = Conv1D(filters=8, kernel_size=3, strides=1, padding='same')(cnn_out2)
    cnn_out2 = LeakyReLU()(cnn_out2)

    cnn_out3 = Conv1D(filters=8, kernel_size=4, strides=1, padding='same')(i)
    cnn_out3 = Conv1D(filters=8, kernel_size=4, strides=1, padding='same')(cnn_out3)
    cnn_out3 = LeakyReLU()(cnn_out3)

    cnn_out = Concatenate(axis=1)([cnn_out1, cnn_out2, cnn_out3])

    cnn_out = Conv1D(filters=4, kernel_size=6, strides=1, padding='same')(cnn_out)
    cnn_out = MaxPool1D(pool_size=2)(cnn_out)

    cnn_out = Flatten()(cnn_out)

    # ############################################

    # lstm_out = LSTM(output_dim=64, return_sequences=True)(i)
    #
    lstm_out = NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1)(i)

    # lstm_out = LSTM(64)(lstm_out)

    # lstm_out = NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1)(lstm_out)
    # lstm_out = Flatten()(lstm_out)

    # 上层叠加新的dense层
    concat_output = Concatenate(axis=1)([cnn_out, lstm_out])
    concat_output = LeakyReLU()(concat_output)

    outputs = Dense(1)(concat_output)

    model = Model(inputs=i, outputs=outputs)
    model.compile(optimizer='rmsprop', loss='mse', )
    model.summary()

    return model


def buildNLSTM_MultiV_v3(timestep):
    batch_size, timesteps, input_dim = None, timestep, 1

    ##############################################################################################

    input_0A3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x00 = NestedLSTM(48, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_0A3)
    x00 = Dense(16, activation='linear')(x00)

    input_0D1 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x01 = NestedLSTM(32, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_0D1)
    x01 = Dense(16, activation='linear')(x01)

    input_0D2 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x02 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_0D2)
    x02 = Dense(16, activation='linear')(x02)

    input_0D3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x03 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_0D3)
    x03 = Dense(16, activation='linear')(x03)

    combined0 = Add()([x00, x01, x02, x03])
    o0 = combined0
    # o0 = Dense(1, activation="linear")(combined0)

    model0 = Model(inputs=[input_0A3, input_0D1, input_0D2, input_0D3], outputs=o0)

    ##############################################################################################

    input_1A3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x10 = NestedLSTM(48, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_1A3)
    x10 = Dense(16, activation='linear')(x10)

    input_1D1 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x11 = NestedLSTM(32, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_1D1)
    x11 = Dense(16, activation='linear')(x11)

    input_1D2 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x12 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_1D2)
    x12 = Dense(16, activation='linear')(x12)

    input_1D3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x13 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_1D3)
    x13 = Dense(16, activation='linear')(x13)

    combined1 = Add()([x10, x11, x12, x13])
    o1 = combined1
    # o1 = Dense(1, activation="linear")(combined1)

    model1 = Model(inputs=[input_1A3, input_1D1, input_1D2, input_1D3], outputs=o1)

    ##############################################################################################

    input_2A3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x20 = NestedLSTM(48, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_2A3)
    x20 = Dense(16, activation='linear')(x20)

    input_2D1 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x21 = NestedLSTM(32, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_2D1)
    x21 = Dense(16, activation='linear')(x21)

    input_2D2 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x22 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_2D2)
    x22 = Dense(16, activation='linear')(x22)

    input_2D3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x23 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_2D3)
    x23 = Dense(16, activation='linear')(x23)

    combined2 = Add()([x20, x21, x22, x23])
    o2 = combined2
    # o2 = Dense(1, activation="linear")(combined2)

    model2 = Model(inputs=[input_2A3, input_2D1, input_2D2, input_2D3], outputs=o2)

    ##############################################################################################

    input_3A3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x30 = NestedLSTM(48, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_3A3)
    x30 = Dense(16, activation='linear')(x30)

    input_3D1 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x31 = NestedLSTM(32, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_3D1)
    x31 = Dense(16, activation='linear')(x31)

    input_3D2 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x32 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_3D2)
    x32 = Dense(16, activation='linear')(x32)

    input_3D3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x33 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_3D3)
    x33 = Dense(16, activation='linear')(x33)

    combined3 = Add()([x30, x31, x32, x33])
    o3 = combined3
    # o3 = Dense(1, activation="linear")(combined3)

    model3 = Model(inputs=[input_3A3, input_3D1, input_3D2, input_3D3], outputs=o3)

    ##############################################################################################

    input_4A3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x40 = NestedLSTM(48, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_4A3)
    x40 = Dense(16, activation='linear')(x40)

    input_4D1 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x41 = NestedLSTM(32, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_4D1)
    x41 = Dense(16, activation='linear')(x41)

    input_4D2 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x42 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_4D2)
    x42 = Dense(16, activation='linear')(x42)

    input_4D3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x43 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_4D3)
    x43 = Dense(16, activation='linear')(x43)

    combined4 = Add()([x40, x41, x42, x43])
    o4 = combined4
    # o4 = Dense(1, activation="linear")(combined4)

    model4 = Model(inputs=[input_4A3, input_4D1, input_4D2, input_4D3], outputs=o4)

    ##############################################################################################

    # combined01 = Concatenate(axis=1)([model0.output,
    #                                   model1.output,
    #                                   model2.output,
    #                                   model3.output,
    #                                   model4.output])
    # Output0 = Dense(1, activation='linear')(combined01)
    #
    # combined11 = Concatenate(axis=1)([model0.output,
    #                                   model1.output,
    #                                   model2.output,
    #                                   model3.output,
    #                                   model4.output])
    # Output1 = Dense(1, activation='linear')(combined11)
    #
    # combined21 = Concatenate(axis=1)([model0.output,
    #                                   model1.output,
    #                                   model2.output,
    #                                   model3.output,
    #                                   model4.output])
    # Output2 = Dense(1, activation='linear')(combined21)
    #
    # combined31 = Concatenate(axis=1)([model0.output,
    #                                   model1.output,
    #                                   model2.output,
    #                                   model3.output,
    #                                   model4.output])
    # Output3 = Dense(1, activation='linear')(combined31)
    #
    # combined41 = Concatenate(axis=1)([model0.output,
    #                                   model1.output,
    #                                   model2.output,
    #                                   model3.output,
    #                                   model4.output])
    # Output4 = Dense(1, activation='linear')(combined41)

    # model = Model(inputs=[model0.input,
    #                       model1.input,
    #                       model2.input,
    #                       model3.input,
    #                       model4.input],
    #               outputs=[Output0,
    #                        Output1,
    #                        Output2,
    #                        Output3,
    #                        Output4])

    combined01 = Concatenate(axis=1)([o0, o1, o2, o3, o4])
    Output0 = Dense(1, activation='linear')(combined01)

    combined11 = Concatenate(axis=1)([o0, o1, o2, o3, o4])
    Output1 = Dense(1, activation='linear')(combined11)

    combined21 = Concatenate(axis=1)([o0, o1, o2, o3, o4])
    Output2 = Dense(1, activation='linear')(combined21)

    combined31 = Concatenate(axis=1)([o0, o1, o2, o3, o4])
    Output3 = Dense(1, activation='linear')(combined31)

    combined41 = Concatenate(axis=1)([o0, o1, o2, o3, o4])
    Output4 = Dense(1, activation='linear')(combined41)

    model = Model(inputs=[input_0A3, input_0D1, input_0D2, input_0D3,
                          input_1A3, input_1D1, input_1D2, input_1D3,
                          input_2A3, input_2D1, input_2D2, input_2D3,
                          input_3A3, input_3D1, input_3D2, input_3D3,
                          input_4A3, input_4D1, input_4D2, input_4D3],
                  outputs=[Output0,
                           Output1,
                           Output2,
                           Output3,
                           Output4])

    model.compile(optimizer='rmsprop', loss='mse', )
    model.summary()

    return model


def buildNLSTM_MultiV_v4(timestep):
    batch_size, timesteps, input_dim = None, timestep, 1

    ##############################################################################################

    input_0A3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x00 = NestedLSTM(48, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_0A3)
    x00 = Dense(16, activation='linear')(x00)

    input_0D1 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x01 = NestedLSTM(32, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_0D1)
    x01 = Dense(16, activation='linear')(x01)

    input_0D2 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x02 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_0D2)
    x02 = Dense(16, activation='linear')(x02)

    input_0D3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x03 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_0D3)
    x03 = Dense(16, activation='linear')(x03)

    combined0 = Add()([x00, x01, x02, x03])
    o0 = combined0
    # o0 = Dense(1, activation="linear")(combined0)

    model0 = Model(inputs=[input_0A3, input_0D1, input_0D2, input_0D3], outputs=o0)

    ##############################################################################################

    input_1A3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x10 = NestedLSTM(48, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_1A3)
    x10 = Dense(16, activation='linear')(x10)

    input_1D1 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x11 = NestedLSTM(32, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_1D1)
    x11 = Dense(16, activation='linear')(x11)

    input_1D2 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x12 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_1D2)
    x12 = Dense(16, activation='linear')(x12)

    input_1D3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x13 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_1D3)
    x13 = Dense(16, activation='linear')(x13)

    combined1 = Add()([x10, x11, x12, x13])
    o1 = combined1
    # o1 = Dense(1, activation="linear")(combined1)

    model1 = Model(inputs=[input_1A3, input_1D1, input_1D2, input_1D3], outputs=o1)

    ##############################################################################################

    input_2A3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x20 = NestedLSTM(48, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_2A3)
    x20 = Dense(16, activation='linear')(x20)

    input_2D1 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x21 = NestedLSTM(32, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_2D1)
    x21 = Dense(16, activation='linear')(x21)

    input_2D2 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x22 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_2D2)
    x22 = Dense(16, activation='linear')(x22)

    input_2D3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x23 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_2D3)
    x23 = Dense(16, activation='linear')(x23)

    combined2 = Add()([x20, x21, x22, x23])
    o2 = combined2
    # o2 = Dense(1, activation="linear")(combined2)

    model2 = Model(inputs=[input_2A3, input_2D1, input_2D2, input_2D3], outputs=o2)

    ##############################################################################################

    input_3A3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x30 = NestedLSTM(48, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_3A3)
    x30 = Dense(16, activation='linear')(x30)

    input_3D1 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x31 = NestedLSTM(32, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_3D1)
    x31 = Dense(16, activation='linear')(x31)

    input_3D2 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x32 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_3D2)
    x32 = Dense(16, activation='linear')(x32)

    input_3D3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x33 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_3D3)
    x33 = Dense(16, activation='linear')(x33)

    combined3 = Add()([x30, x31, x32, x33])
    o3 = combined3
    # o3 = Dense(1, activation="linear")(combined3)

    model3 = Model(inputs=[input_3A3, input_3D1, input_3D2, input_3D3], outputs=o3)

    ##############################################################################################

    input_4A3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x40 = NestedLSTM(48, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_4A3)
    x40 = Dense(16, activation='linear')(x40)

    input_4D1 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x41 = NestedLSTM(32, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_4D1)
    x41 = Dense(16, activation='linear')(x41)

    input_4D2 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x42 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_4D2)
    x42 = Dense(16, activation='linear')(x42)

    input_4D3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x43 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_4D3)
    x43 = Dense(16, activation='linear')(x43)

    combined4 = Add()([x40, x41, x42, x43])
    o4 = combined4
    # o4 = Dense(1, activation="linear")(combined4)

    model4 = Model(inputs=[input_4A3, input_4D1, input_4D2, input_4D3], outputs=o4)

    ##############################################################################################

    input_5A3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x50 = NestedLSTM(48, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_5A3)
    x50 = Dense(16, activation='linear')(x50)

    input_5D1 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x51 = NestedLSTM(32, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_5D1)
    x51 = Dense(16, activation='linear')(x51)

    input_5D2 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x52 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_5D2)
    x52 = Dense(16, activation='linear')(x52)

    input_5D3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x53 = NestedLSTM(24, depth=2, dropout=0.0, recurrent_dropout=0.1)(input_5D3)
    x53 = Dense(16, activation='linear')(x53)

    combined5 = Add()([x50, x51, x52, x53])
    o5 = combined5
    # o4 = Dense(1, activation="linear")(combined4)

    model5 = Model(inputs=[input_5A3, input_5D1, input_5D2, input_5D3], outputs=o5)

    ##############################################################################################

    combined = Concatenate(axis=1)([o0, o1, o2, o3, o4, o5])

    Output0 = Dense(1, activation='linear')(combined)

    Output1 = Dense(1, activation='linear')(combined)

    Output2 = Dense(1, activation='linear')(combined)

    Output3 = Dense(1, activation='linear')(combined)

    Output4 = Dense(1, activation='linear')(combined)

    Output5 = Dense(1, activation='linear')(combined)

    model = Model(inputs=[input_0A3, input_0D1, input_0D2, input_0D3,
                          input_1A3, input_1D1, input_1D2, input_1D3,
                          input_2A3, input_2D1, input_2D2, input_2D3,
                          input_3A3, input_3D1, input_3D2, input_3D3,
                          input_4A3, input_4D1, input_4D2, input_4D3,
                          input_5A3, input_5D1, input_5D2, input_5D3],
                  outputs=[Output0,
                           Output1,
                           Output2,
                           Output3,
                           Output4,
                           Output5])

    model.compile(optimizer='rmsprop', loss='mse', )
    model.summary()

    return model