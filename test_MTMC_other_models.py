# from matplotlib import pyplot as plt
# import os
# import openpyxl
# import pandas as pd
# import numpy as np
import csv
import warnings
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPRegressor  #### MLP 感知机 ####
from sklearn.tree import ExtraTreeRegressor  #### ExtraTree 极端随机树回归####
from sklearn import tree  #### 决策树回归 ####
from sklearn.ensemble import BaggingRegressor  #### Bagging回归 ####
from sklearn.ensemble import AdaBoostRegressor  #### Adaboost回归
from sklearn import linear_model  #### 线性回归####
from sklearn import svm  #### SVM回归####
from sklearn import ensemble  #### Adaboost回归####  ####3.7GBRT回归####  ####3.5随机森林回归####
from sklearn import neighbors  #### KNN回归####

from Model.model_major import *
# from model_lstm import lstmRegressor
# from model_tcn import TCN_LSTM
# from support_nested_lstm import NestedLSTM
# from model_slstm import SLSTM

from pyhht.emd import EMD
from Support.support_wavelet import *
from Support.support_VMD import VMD

from Part.part_evaluate import *
from Part.part_data_preprocessing import *
from test_MTMC import neo_prediction

warnings.filterwarnings("ignore")


def pre_model(model, trainX, trainY, testX):
    model.fit(trainX, trainY)

    predict = model.predict(testX)
    return predict


########################################################################


def load_data_ts(trainNum, testNum, startNum, data):
    print('General_data loading.')

    global ahead_num
    # all_data_checked = data

    targetData = data

    # 处理预测信息，划分训练集和测试集
    # PM = targetData[startNum : startNum + trainNum + testNum]
    # PM = np.array(PM).reshape(-1, 1)
    targetData = targetData[startNum + 1: startNum + trainNum + testNum + 1]
    targetData = np.array(targetData).reshape(-1, 1)

    # #归一化，每个特征分开归一化
    global scaler_target
    scaler_target = StandardScaler(copy=True, with_mean=True, with_std=True)
    targetData = scaler_target.fit_transform(targetData)
    # print("targetData:", targetData.shape)

    global x_mode

    time_series_y = create_time_series(targetData, ahead_num)
    allX = np.c_[time_series_y]
    allX = allX.T

    ###########======================================

    trainX = allX[:, : trainNum]
    trainY = targetData.T[:, ahead_num: trainNum + ahead_num]
    testX = allX[:, trainNum:]
    testY = targetData.T[:, trainNum + ahead_num: (trainNum + testNum)]

    trainY = trainY.flatten()  # 降维
    testY = testY.flatten()  # 降维
    trainX = trainX.T
    testX = testX.T

    print('load_data complete.\n')

    return trainX, trainY, testX, testY


def load_data_emd(trainNum, testNum, startNum, data):
    print('EMD_data loading.')

    global ahead_num
    # all_data_checked = data

    targetData = data

    # 处理预测信息，划分训练集和测试集
    targetData = targetData[startNum + 1: startNum + trainNum + testNum + 1]
    targetData = np.array(targetData).reshape(-1, 1)

    # #归一化，每个特征分开归一化
    global scaler_target
    scaler_target = StandardScaler(copy=True, with_mean=True, with_std=True)
    targetData = scaler_target.fit_transform(targetData)

    decomposer = EMD(targetData)
    imfs = decomposer.decompose()
    # plot_imfs(targetData, imfs)
    data_decomposed = imfs.tolist()

    for h1 in range(len(data_decomposed)):
        data_decomposed[h1] = np.array(data_decomposed[h1]).reshape(-1, 1)
    for h2 in range(len(data_decomposed)):
        trainX, trainY, testX, testY = create_data(data_decomposed[h2], trainNum, ahead_num)
        dataset_imf = [trainX, trainY, testX, testY]
        data_decomposed[h2] = dataset_imf

    print('load_data complete.\n')

    return data_decomposed


def load_data_wvlt(trainNum, testNum, startNum, data):
    print('wavelet_data loading.')

    global ahead_num
    # all_data_checked = data
    targetData = data

    # 处理预测信息，划分训练集和测试集
    targetData = targetData[startNum + 1: startNum + trainNum + testNum + 1]
    targetData = np.array(targetData).reshape(-1, 1)

    # #归一化，每个特征分开归一化
    global scaler_target
    scaler_target = StandardScaler(copy=True, with_mean=True, with_std=True)
    targetData = scaler_target.fit_transform(targetData)

    testY = targetData[trainNum: (trainNum + testNum), :]
    wavefun = pywt.Wavelet('db1')
    global wvlt_lv

    coeffs = swt_decom(targetData, wavefun, wvlt_lv)

    ### 测试滤波效果
    wvlt_level_list = []
    for wvlt_level in range(len(coeffs)):
        wvlt_trainX, wvlt_trainY, wvlt_testX, wvlt_testY = create_data(coeffs[wvlt_level], trainNum, ahead_num)
        wvlt_level_part = [wvlt_trainX, wvlt_trainY, wvlt_testX, wvlt_testY]
        wvlt_level_list.append(wvlt_level_part)

    print('load_data complete.\n')

    return wvlt_level_list, testY


def load_data_VMD(trainNum, testNum, startNum, data):
    print('VMD_data loading.')

    global ahead_num
    # all_data_checked = data
    targetData = data

    # 处理预测信息，划分训练集和测试集
    targetData = targetData[startNum + 1: startNum + trainNum + testNum + 1]
    targetData = np.array(targetData).reshape(-1, 1)

    # #归一化，每个特征分开归一化
    global scaler_target
    scaler_target = StandardScaler(copy=True, with_mean=True, with_std=True)
    targetData = scaler_target.fit_transform(targetData)

    testY = targetData[trainNum: (trainNum + testNum), :]

    VMD_level = wvlt_lv + 1
    imf_list = VMD(targetData.reshape(-1, ), VMD_level)
    imf_list = imf_list.tolist()

    coeffs = []
    for i in range(len(imf_list)):
        imf = imf_list[i]
        for j in range(len(imf)):
            part_real = imf[j].real
            imf[j] = part_real
        coeffs.append(np.array(imf).reshape(-1, 1))

    coeffs_rest = 0
    for i in range(len(coeffs)):
        coeffs_rest = coeffs_rest + coeffs[i]
    coeffs_rest = targetData - coeffs_rest

    coeffs[0] = coeffs[0] + coeffs[1]
    coeffs[1] = coeffs_rest

    # imf_list = VMD(coeffs_rest.reshape(-1, ), VMD_level)
    # imf_list = imf_list.tolist()
    #
    # coeffs = []
    # for i in range(len(imf_list)):
    #     imf = imf_list[i]
    #     for j in range(len(imf)):
    #         part_real = imf[j].real
    #         imf[j] = part_real
    #     coeffs.append(np.array(imf).reshape(-1, 1))

    # plt.figure(figsize=(19, 8))
    # plt.subplot(611)
    # plt.plot(targetData[:1000, :])
    # plt.subplot(612)
    # plt.plot(coeffs[0][:1000, :])
    # plt.subplot(613)
    # plt.plot(coeffs[1][:1000, :])
    # plt.subplot(614)
    # plt.plot(coeffs[2][:1000, :])
    # plt.subplot(615)
    # plt.plot(coeffs[3][:1000, :])
    # plt.subplot(616)
    # plt.plot(coeffs_rest[:1000, :])
    # plt.subplots_adjust(left=0.04, bottom=0.05, right=0.99, top=0.95, hspace=0.25)
    # plt.show()

    ### 测试滤波效果
    decomposed_data = []
    for i in range(len(coeffs)):
        VMD_trainX, VMD_trainY, VMD_testX, VMD_testY = create_data(coeffs[i], trainNum, ahead_num)
        decomposed_data.append([VMD_trainX, VMD_trainY, VMD_testX, VMD_testY])

    print('load_data complete.\n')

    return decomposed_data, testY


#########################################################################


def Decide_Tree(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'decideTree'
    model_name_short = 'dTr'
    print(model_name + ' Start.')

    model_DecisionTreeRegressor = tree.DecisionTreeRegressor()  # 决策树
    predict_decideTree = pre_model(model_DecisionTreeRegressor, x_train, y_train, x_test)
    predict_decideTree = scaler_target.inverse_transform(predict_decideTree)
    flag_decideTree = deal_flag(predict_decideTree, minLen)
    accuracy_decideTree = deal_accuracy(y_rampflag, flag_decideTree)

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name, model_name_short, dataY, predict_decideTree, accuracy_decideTree)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_decideTree


def Random_forest(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'randomForest'
    model_name_short = 'rdF'
    print(model_name + ' Start.')

    model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=50)  # 随机森林
    predict_randomForest = pre_model(model_RandomForestRegressor, x_train, y_train, x_test)
    predict_randomForest = scaler_target.inverse_transform(predict_randomForest)
    flag_randomForest = deal_flag(predict_randomForest, minLen)
    accuracy_randomForest = deal_accuracy(y_rampflag, flag_randomForest)

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name, model_name_short, dataY, predict_randomForest,
                                        accuracy_randomForest)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_randomForest


def SVR(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'svr'
    model_name_short = 'svr'
    print(model_name + ' Start.')

    model_SVR = svm.SVR()  # SVR回归
    predict_svr = pre_model(model_SVR, x_train, y_train, x_test)
    predict_svr = scaler_target.inverse_transform(predict_svr)
    flag_svr = deal_flag(predict_svr, minLen)
    accuracy_svr = deal_accuracy(y_rampflag, flag_svr)

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name, model_name_short, dataY, predict_svr, accuracy_svr)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_svr


def MLP(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'mlp'
    model_name_short = 'mlp'
    print(model_name + ' Start.')

    model_MLP = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(20, 20, 20), random_state=2)  # MLP
    predict_mlp = pre_model(model_MLP, x_train, y_train, x_test)
    predict_mlp = scaler_target.inverse_transform(predict_mlp)
    flag_mlp = deal_flag(predict_mlp, minLen)
    accuracy_mlp = deal_accuracy(y_rampflag, flag_mlp)

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name,
                                        model_name_short,
                                        dataY,
                                        predict_mlp,
                                        accuracy_mlp)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_mlp


#########################################################################


def LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'LSTM'
    model_name_short = 'LSTM'
    print(model_name + ' Start.')

    feature_num = x_train.shape[1]
    x_train_lstm = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test_lstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Modelling and Predicting
    print('Build model...')

    model = build_LSTM(feature_num)
    # model = buildOLSTM(feature_num)

    global show_process
    time_callback = TimeHistory()
    history = model.fit(x_train_lstm, y_train, epochs=16, validation_split=0.05, verbose=show_process,
                        callbacks=[time_callback])
    train_time = time_callback.totaltime

    predict_lstm = model.predict(x_test_lstm)

    predict_lstm = predict_lstm.reshape(-1, )
    predict_lstm = scaler_target.inverse_transform(predict_lstm)
    flag_lstm = deal_flag(predict_lstm, minLen)
    accuracy_lstm = deal_accuracy(y_rampflag, flag_lstm)

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_lstm,
                                           accuracy_lstm,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_lstm


def EMD_LSTM(emd_list, y_rampflag, dataY, emd_decay_rate):
    model_name = 'EMD'
    model_name_short = 'EMD'
    print(model_name + ' Start.')

    model = build_LSTM(ahead_num)

    global show_process

    predict_emd_list = []
    train_time_total = 0
    for ie in range(len(emd_list)):
        # emd_list[ie][0] = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        # emd_list[ie][2] = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trX = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        EMD_teX = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trY = emd_list[ie][1]

        time_callback = TimeHistory()
        model.fit(EMD_trX, EMD_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time
        # model.fit(emd_list[ie][0], emd_list[ie][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        predict_emd_part = model.predict(EMD_teX)
        predict_emd_part = predict_emd_part.reshape(-1, )
        predict_emd_list.append(predict_emd_part)
        print('EMD_imf ' + str(ie + 1) + ' Complete.')

    # wavefun = pywt.Wavelet('db1')
    # predict_EN = iswt_decom(predict_emd_list, wavefun)

    predict_EMD = predict_emd_list[-1]
    for i in range(len(predict_emd_list) - 1):
        predict_EMD = predict_EMD + predict_emd_list[i] * emd_decay_rate
    # predict_noise = predict_EN - predict_emd_list[-1]

    predict_EMD = scaler_target.inverse_transform(predict_EMD)
    flag_EMD = deal_flag(predict_EMD, minLen)
    accuracy_EMD = deal_accuracy(y_rampflag, flag_EMD)

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_EMD,
                                           accuracy_EMD,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_EMD


def Wavelet_LSTM(wvlt_list, y_rampflag, dataY):
    model_name = 'Wvlt'
    model_name_short = 'Wvlt'
    print(model_name + ' Start.')

    model = build_LSTM(ahead_num)

    global show_process

    train_time_total = 0
    predict_Wvlt_list = []
    for i_wvlt in range(len(wvlt_list)):
        wvlt_trX = np.reshape(wvlt_list[i_wvlt][0],
                              (wvlt_list[i_wvlt][0].shape[0],
                               wvlt_list[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list[i_wvlt][2],
                              (wvlt_list[i_wvlt][2].shape[0],
                               wvlt_list[i_wvlt][2].shape[1], 1))
        wvlt_trY = wvlt_list[i_wvlt][1]

        time_callback = TimeHistory()
        model.fit(wvlt_trX, wvlt_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        # model.fit(wvlt_list[i_wvlt][0], wvlt_list[i_wvlt][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time

        predict_Wvlt = model.predict(wvlt_teX)
        predict_Wvlt = predict_Wvlt.reshape(-1, )
        predict_Wvlt_list.append(predict_Wvlt)
        print('wvlt_level ' + str(i_wvlt + 1) + ' Complete.')

    wavefun = pywt.Wavelet('db1')

    predict_Wvlt = iswt_decom(predict_Wvlt_list, wavefun)
    predict_Wvlt = scaler_target.inverse_transform(predict_Wvlt)
    flag_Wvlt = deal_flag(predict_Wvlt, minLen)
    accuracy_Wvlt = deal_accuracy(y_rampflag, flag_Wvlt)

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_Wvlt,
                                           accuracy_Wvlt,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_Wvlt


def VMD_LSTM(data_list, y_rampflag, dataY):
    model_name = 'VMD'
    model_name_short = 'VMD'
    print(model_name + ' Start.')

    model = buildNLSTM(ahead_num)

    global show_process

    train_time_total = 0
    predict_VMD_list = []
    for i_vmd in range(len(data_list)):
        VMD_trX = np.reshape(data_list[i_vmd][0],
                             (data_list[i_vmd][0].shape[0],
                              data_list[i_vmd][0].shape[1], 1))
        VMD_teX = np.reshape(data_list[i_vmd][2],
                             (data_list[i_vmd][2].shape[0],
                              data_list[i_vmd][2].shape[1], 1))
        VMD_trY = data_list[i_vmd][1]

        time_callback = TimeHistory()
        model.fit(VMD_trX, VMD_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])

        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time

        predict_VMD = model.predict(VMD_teX)
        predict_VMD = predict_VMD.reshape(-1, )
        predict_VMD_list.append(predict_VMD)
        print('VMD_level ' + str(i_vmd + 1) + ' Complete.')

    predict_VMD_list = np.array(predict_VMD_list).T
    predict_VMD_list = predict_VMD_list.tolist()
    for j_vmd in range(len(predict_VMD_list)):
        predict_VMD_list[j_vmd] = sum(predict_VMD_list[j_vmd])
    predict_VMD = np.array(predict_VMD_list).reshape(-1, )

    predict_VMD = scaler_target.inverse_transform(predict_VMD)
    flag_VMD = deal_flag(predict_VMD, minLen)
    accuracy_VMD = deal_accuracy(y_rampflag, flag_VMD)

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_VMD,
                                           accuracy_VMD,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_VMD


#########################################################################


def Nested_LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'NLSTM'
    model_name_short = 'NLSTM'
    print('NLSTM Begin.')

    feature_num = x_train.shape[1]
    x_train_nlstm = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test_nlstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Modelling and Predicting
    print('Build model...')

    model = buildNLSTM(feature_num)
    # model = buildOLSTM(feature_num)

    global show_process

    time_callback = TimeHistory()
    history = model.fit(x_train_nlstm, y_train, epochs=16, validation_split=0.05, verbose=show_process,
                        callbacks=[time_callback])
    train_time = time_callback.totaltime

    predict_nlstm = model.predict(x_test_nlstm)

    # history = model.fit(x_train, y_train, epochs=16, validation_split=0.05, verbose=2)
    #
    # predict_nlstm = model.predict(x_test)
    predict_nlstm = predict_nlstm.reshape(-1, )
    predict_nlstm = scaler_target.inverse_transform(predict_nlstm)
    flag_nlstm = deal_flag(predict_nlstm, minLen)
    accuracy_nlstm = deal_accuracy(y_rampflag, flag_nlstm)

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_nlstm,
                                           accuracy_nlstm,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    print('NLSTM Complete.')
    return predict_nlstm


def EMD_NLSTM(emd_list, y_rampflag, dataY, emd_decay_rate):
    model_name = 'EN'
    model_name_short = 'EN'
    print(model_name + ' Start.')

    model = buildNLSTM(ahead_num)

    global show_process

    predict_emd_list = []
    train_time_total = 0
    for ie in range(len(emd_list)):
        # emd_list[ie][0] = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        # emd_list[ie][2] = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trX = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        EMD_teX = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trY = emd_list[ie][1]
        time_callback = TimeHistory()
        model.fit(EMD_trX, EMD_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time
        # model.fit(emd_list[ie][0], emd_list[ie][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        predict_emd_part = model.predict(EMD_teX)
        predict_emd_part = predict_emd_part.reshape(-1, )
        predict_emd_list.append(predict_emd_part)
        print('EMD_imf ' + str(ie + 1) + ' Complete.')

    # wavefun = pywt.Wavelet('db1')
    # predict_EN = iswt_decom(predict_emd_list, wavefun)

    predict_EN = predict_emd_list[-1]
    for i in range(len(predict_emd_list) - 1):
        predict_EN = predict_EN + predict_emd_list[i] * emd_decay_rate
    # predict_noise = predict_EN - predict_emd_list[-1]

    predict_EN = scaler_target.inverse_transform(predict_EN)
    flag_EN = deal_flag(predict_EN, minLen)
    accuracy_EN = deal_accuracy(y_rampflag, flag_EN)

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_EN,
                                           accuracy_EN,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_EN


def Wavelet_NLSTM(wvlt_list, y_rampflag, dataY):
    model_name = 'WN'
    model_name_short = 'WN'
    print(model_name + ' Start.')

    model = buildNLSTM(ahead_num)

    global show_process

    train_time_total = 0
    predict_WN_list = []
    for i_wvlt in range(len(wvlt_list)):
        wvlt_trX = np.reshape(wvlt_list[i_wvlt][0],
                              (wvlt_list[i_wvlt][0].shape[0],
                               wvlt_list[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list[i_wvlt][2],
                              (wvlt_list[i_wvlt][2].shape[0],
                               wvlt_list[i_wvlt][2].shape[1], 1))
        wvlt_trY = wvlt_list[i_wvlt][1]

        time_callback = TimeHistory()
        model.fit(wvlt_trX, wvlt_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        # model.fit(wvlt_list[i_wvlt][0], wvlt_list[i_wvlt][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time

        predict_WN = model.predict(wvlt_teX)
        predict_WN = predict_WN.reshape(-1, )
        predict_WN_list.append(predict_WN)
        print('wvlt_level ' + str(i_wvlt + 1) + ' Complete.')

    wavefun = pywt.Wavelet('db1')

    predict_WN = iswt_decom(predict_WN_list, wavefun)
    predict_WN = scaler_target.inverse_transform(predict_WN)
    flag_WN = deal_flag(predict_WN, minLen)
    accuracy_WN = deal_accuracy(y_rampflag, flag_WN)

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_WN,
                                           accuracy_WN,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_WN


def VMD_NLSTM(data_list, y_rampflag, dataY):
    model_name = 'VN'
    model_name_short = 'VN'
    print(model_name + ' Start.')

    model = buildNLSTM(ahead_num)

    global show_process

    train_time_total = 0
    predict_VN_list = []
    for i_vmd in range(len(data_list)):
        VMD_trX = np.reshape(data_list[i_vmd][0],
                             (data_list[i_vmd][0].shape[0],
                              data_list[i_vmd][0].shape[1], 1))
        VMD_teX = np.reshape(data_list[i_vmd][2],
                             (data_list[i_vmd][2].shape[0],
                              data_list[i_vmd][2].shape[1], 1))
        VMD_trY = data_list[i_vmd][1]

        time_callback = TimeHistory()
        model.fit(VMD_trX, VMD_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])

        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time

        predict_VN = model.predict(VMD_teX)
        predict_VN = predict_VN.reshape(-1, )
        predict_VN_list.append(predict_VN)
        print('VMD_level ' + str(i_vmd + 1) + ' Complete.')

    predict_VN_list = np.array(predict_VN_list).T
    predict_VN_list = predict_VN_list.tolist()
    for j_vmd in range(len(predict_VN_list)):
        predict_VN_list[j_vmd] = sum(predict_VN_list[j_vmd])
    predict_VN = np.array(predict_VN_list).reshape(-1, )

    predict_VN = scaler_target.inverse_transform(predict_VN)
    flag_VN = deal_flag(predict_VN, minLen)
    accuracy_VN = deal_accuracy(y_rampflag, flag_VN)

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_VN,
                                           accuracy_VN,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_VN


#########################################################################


def Stacked_LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'SLSTM'
    model_name_short = 'SLSTM'
    print(model_name + ' Start.')

    feature_num = x_train.shape[1]
    x_train_slstm = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test_slstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Modelling and Predicting
    print('Build model...')

    model = buildSLSTM(feature_num)
    # model = buildOLSTM(feature_num)

    global show_process

    time_callback = TimeHistory()
    history = model.fit(x_train_slstm, y_train, epochs=16, validation_split=0.05, verbose=show_process,
                        callbacks=[time_callback])
    train_time = time_callback.totaltime

    predict_slstm = model.predict(x_test_slstm)

    # history = model.fit(x_train, y_train, epochs=16, validation_split=0.05, verbose=2)
    #
    # predict_slstm = model.predict(x_test)
    predict_slstm = predict_slstm.reshape(-1, )
    predict_slstm = scaler_target.inverse_transform(predict_slstm)
    flag_slstm = deal_flag(predict_slstm, minLen)
    accuracy_slstm = deal_accuracy(y_rampflag, flag_slstm)

    # rmse_slstm = RMSE1(dataY, predict_slstm)
    # mape_slstm = MAPE1(dataY, predict_slstm)
    # mae_slstm = MAE1(dataY, predict_slstm)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_slstm: {}'.format(mae_slstm)
    # eva_output += '\nrmse_slstm: {}'.format(rmse_slstm)
    # eva_output += '\nmape_slstm: {}'.format(mape_slstm)
    # eva_output += '\naccuracy_slstm: {}'.format(accuracy_slstm)
    # result_all.append(['slstm', mae_slstm, rmse_slstm, mape_slstm, accuracy_slstm])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_slstm,
                                           accuracy_slstm,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_slstm


def EMD_SLSTM(emd_list, y_rampflag, dataY, emd_decay_rate):
    model_name = 'ES'
    model_name_short = 'ES'
    print(model_name + ' Start.')

    model = buildSLSTM(ahead_num)

    global show_process

    predict_emd_list = []
    train_time_total = 0
    for ie in range(len(emd_list)):
        # emd_list[ie][0] = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        # emd_list[ie][2] = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trX = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        EMD_teX = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trY = emd_list[ie][1]
        time_callback = TimeHistory()
        model.fit(EMD_trX, EMD_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time
        # model.fit(emd_list[ie][0], emd_list[ie][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        predict_emd_part = model.predict(EMD_teX)
        predict_emd_part = predict_emd_part.reshape(-1, )
        predict_emd_list.append(predict_emd_part)
        print('EMD_imf ' + str(ie + 1) + ' Complete.')

    # wavefun = pywt.Wavelet('db1')
    # predict_ES = iswt_decom(predict_emd_list, wavefun)

    predict_ES = predict_emd_list[-1]
    for i in range(len(predict_emd_list) - 1):
        predict_ES = predict_ES + predict_emd_list[i] * emd_decay_rate
    # predict_noise = predict_ES - predict_emd_list[-1]

    predict_ES = scaler_target.inverse_transform(predict_ES)
    flag_ES = deal_flag(predict_ES, minLen)
    accuracy_ES = deal_accuracy(y_rampflag, flag_ES)

    # rmse_ES = RMSE1(dataY, predict_ES)
    # mape_ES = MAPE1(dataY, predict_ES)
    # mae_ES = MAE1(dataY, predict_ES)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_ES: {}'.format(mae_ES)
    # eva_output += '\nrmse_ES: {}'.format(rmse_ES)
    # eva_output += '\nmape_ES: {}'.format(mape_ES)
    # eva_output += '\naccuracy_ES: {}'.format(accuracy_ES)
    # result_all.append(['EN', mae_ES, rmse_ES, mape_ES, accuracy_ES])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_ES,
                                           accuracy_ES,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_ES


def Wavelet_SLSTM(wvlt_list, y_rampflag, dataY):
    model_name = 'WS'
    model_name_short = 'WS'
    print(model_name + ' Start.')

    model = buildSLSTM(ahead_num)

    global show_process

    train_time_total = 0
    predict_WS_list = []
    for i_wvlt in range(len(wvlt_list)):
        wvlt_trX = np.reshape(wvlt_list[i_wvlt][0],
                              (wvlt_list[i_wvlt][0].shape[0],
                               wvlt_list[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list[i_wvlt][2],
                              (wvlt_list[i_wvlt][2].shape[0],
                               wvlt_list[i_wvlt][2].shape[1], 1))
        wvlt_trY = wvlt_list[i_wvlt][1]

        time_callback = TimeHistory()
        model.fit(wvlt_trX, wvlt_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        # model.fit(wvlt_list[i_wvlt][0], wvlt_list[i_wvlt][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time

        predict_WS = model.predict(wvlt_teX)
        predict_WS = predict_WS.reshape(-1, )
        predict_WS_list.append(predict_WS)
        print('wvlt_level ' + str(i_wvlt + 1) + ' Complete.')

    wavefun = pywt.Wavelet('db1')

    predict_WS = iswt_decom(predict_WS_list, wavefun)
    predict_WS = scaler_target.inverse_transform(predict_WS)
    flag_WS = deal_flag(predict_WS, minLen)
    accuracy_WS = deal_accuracy(y_rampflag, flag_WS)

    # # dataY1 = scaler_target.inverse_transform(dataY1)
    # # dataY1 = dataY1.T.tolist()[0]
    # rmse_WS = RMSE1(dataY, predict_WS)
    # mape_WS = MAPE1(dataY, predict_WS)
    # mae_WS = MAE1(dataY, predict_WS)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_WS: {}'.format(mae_WS)
    # eva_output += '\nrmse_WS: {}'.format(rmse_WS)
    # eva_output += '\nmape_WS: {}'.format(mape_WS)
    # eva_output += '\naccuracy_WS: {}'.format(accuracy_WS)
    # result_all.append(['WN', mae_WS, rmse_WS, mape_WS, accuracy_WS])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_WS,
                                           accuracy_WS,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_WS


def VMD_SLSTM(data_list, y_rampflag, dataY):
    model_name = 'VS'
    model_name_short = 'VS'
    print(model_name + ' Start.')

    model = buildSLSTM(ahead_num)

    global show_process

    train_time_total = 0
    predict_VS_list = []
    for i_vmd in range(len(data_list)):
        VMD_trX = np.reshape(data_list[i_vmd][0],
                             (data_list[i_vmd][0].shape[0],
                              data_list[i_vmd][0].shape[1], 1))
        VMD_teX = np.reshape(data_list[i_vmd][2],
                             (data_list[i_vmd][2].shape[0],
                              data_list[i_vmd][2].shape[1], 1))
        VMD_trY = data_list[i_vmd][1]

        time_callback = TimeHistory()
        model.fit(VMD_trX, VMD_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])

        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time

        predict_VS = model.predict(VMD_teX)
        predict_VS = predict_VS.reshape(-1, )
        predict_VS_list.append(predict_VS)
        print('VMD_level ' + str(i_vmd + 1) + ' Complete.')

    predict_VS_list = np.array(predict_VS_list).T
    predict_VS_list = predict_VS_list.tolist()
    for j_vmd in range(len(predict_VS_list)):
        predict_VS_list[j_vmd] = sum(predict_VS_list[j_vmd])
    predict_VS = np.array(predict_VS_list).reshape(-1, )

    predict_VS = scaler_target.inverse_transform(predict_VS)
    flag_VS = deal_flag(predict_VS, minLen)
    accuracy_VS = deal_accuracy(y_rampflag, flag_VS)

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_VS,
                                           accuracy_VS,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_VS


#########################################################################


def main(start_num, interval_ori, order):
    # random_state = np.random.RandomState(7)
    np.random.RandomState(7)

    #########################################################################

    # lookback number
    global ahead_num
    ahead_num = 4

    global interval
    interval = interval_ori

    global minLen
    minLen = 0

    # 1 for features; 2 for timeseries; 3 for feature & timeseries.
    global x_mode
    x_mode = 2

    global wvlt_lv
    wvlt_lv = 3

    global show_process
    show_process = 1

    #########################################################################

    num = 12
    filename1 = "dataset\\PRSA_Data_"
    filename2 = ".csv"
    filename = [filename1, filename2]

    # training number
    startNum = start_num
    trainNum = (24 * 1000) // interval
    testNum = ((24 * 100) // interval) + ahead_num

    emd_decay_rate = 1.00

    #########################################################################

    if order == 'PM2':
        dataset = read_csv_PM2(filename, trainNum, testNum, startNum, num, interval)
    elif order == 'PM10':
        dataset = read_csv_PM10(filename, trainNum, testNum, startNum, num, interval)
    elif order == 'SO2':
        dataset = read_csv_SO2(filename, trainNum, testNum, startNum, num, interval)
    elif order == 'NO2':
        dataset = read_csv_NO2(filename, trainNum, testNum, startNum, num, interval)
    elif order == 'CO':
        dataset = read_csv_CO(filename, trainNum, testNum, startNum, num, interval)
    elif order == 'O3':
        dataset = read_csv_O3(filename, trainNum, testNum, startNum, num, interval)
    else:
        dataset = read_csv_PM2(filename, trainNum, testNum, startNum, num, interval)
        order = 'PM2'
        print('Unknown Order, PM2 predicted as default.')

    #########################################################################

    x_train, y_train, x_test, y_test = load_data_ts(trainNum, testNum, startNum, dataset)
    emd_list = load_data_emd(trainNum, testNum, startNum, dataset)
    wvlt_list, _ = load_data_wvlt(trainNum, testNum, startNum, dataset)
    VMD_list, _ = load_data_VMD(trainNum, testNum, startNum, dataset)

    #########################################################################

    # #####culculate Accuracy by rampflag
    global scaler_target
    dataY = scaler_target.inverse_transform(y_test)
    # minLen = np.mean(dataY) * 0.25
    minLen = 0
    print('Accuracy Flag:', minLen)
    y_rampflag = deal_flag(dataY, minLen)

    ######=========================Modelling and Predicting=========================#####
    print("======================================================")
    global eva_output, result_all
    eva_output = '\nEvaluation.'
    result_all = []

    Dict_Prediction = {}

    predict_decideTree = Decide_Tree(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    Dict_Prediction['predict_decideTree'] = predict_decideTree
    predict_randomForest = Random_forest(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    Dict_Prediction['predict_randomForest'] = predict_randomForest
    predict_svr = SVR(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    Dict_Prediction['predict_svr'] = predict_svr
    predict_mlp = MLP(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    Dict_Prediction['predict_mlp'] = predict_mlp

    predict_lstm = LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    Dict_Prediction['predict_lstm'] = predict_lstm
    predict_lstm_emd = EMD_LSTM(emd_list, y_rampflag, dataY, emd_decay_rate)
    Dict_Prediction['predict_lstm_emd'] = predict_lstm_emd
    predict_wavelet = Wavelet_LSTM(wvlt_list, y_rampflag, dataY)
    Dict_Prediction['predict_wavelet'] = predict_wavelet
    predict_VMD = VMD_LSTM(VMD_list, y_rampflag, dataY)
    Dict_Prediction['predict_VMD'] = predict_VMD

    predict_nlstm = Nested_LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    Dict_Prediction['predict_nlstm'] = predict_nlstm
    predict_EN = EMD_NLSTM(emd_list, y_rampflag, dataY, emd_decay_rate)
    Dict_Prediction['predict_EN'] = predict_EN
    predict_WN = Wavelet_NLSTM(wvlt_list, y_rampflag, dataY)
    Dict_Prediction['predict_WN'] = predict_WN
    predict_VN = VMD_NLSTM(VMD_list, y_rampflag, dataY)
    Dict_Prediction['predict_VN'] = predict_VN

    predict_slstm = Stacked_LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    Dict_Prediction['predict_slstm'] = predict_slstm
    predict_ES = EMD_SLSTM(emd_list, y_rampflag, dataY, emd_decay_rate)
    Dict_Prediction['predict_ES'] = predict_ES
    predict_WS = Wavelet_SLSTM(wvlt_list, y_rampflag, dataY)
    Dict_Prediction['predict_WS'] = predict_WS
    predict_VS = VMD_SLSTM(VMD_list, y_rampflag, dataY)
    Dict_Prediction['predict_VS'] = predict_VS

    print(eva_output)
    Dict_Prediction['result_all'] = result_all
    Dict_Prediction['Name'] = order

    np.save('saved\\prediction_ts4_otherbase_'+order+'.npy', Dict_Prediction)

    save_file_name = "result\\result_"+str(order)+".csv"

    csv_file = open(save_file_name, "w", newline="")  # 创建csv文件
    writer = csv.writer(csv_file)  # 创建写的对象
    writer.writerow(["", "MAE", "RMSE", "MAPE", "R2", "ACC", "TIME"])
    for wtr in range(len(result_all)):
        writer.writerow(result_all[wtr])

    # backup for Denied permission.
    save_file_name = "result\\result_"+str(order)+"_backup.csv"

    csv_file = open(save_file_name, "w", newline="")  # 创建csv文件
    writer = csv.writer(csv_file)  # 创建写的对象
    writer.writerow(["", "MAE", "RMSE", "MAPE", "R2", "ACC", "TIME"])
    for wtr in range(len(result_all)):
        writer.writerow(result_all[wtr])

    return result_all


if __name__ == "__main__":

    start_num = 0
    interval_ori = 1
    _ = neo_prediction(start_num, interval_ori)
    _ = main(start_num, interval_ori, order='PM2')
    _ = main(start_num, interval_ori, order='PM10')
    _ = main(start_num, interval_ori, order='SO2')
    _ = main(start_num, interval_ori, order='NO2')
    _ = main(start_num, interval_ori, order='CO')
    _ = main(start_num, interval_ori, order='O3')
