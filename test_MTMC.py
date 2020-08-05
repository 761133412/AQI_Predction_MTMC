
# from matplotlib import pyplot as plt
# import os
# import openpyxl
# import pandas as pd
# import numpy as np
import csv
import warnings
from sklearn.preprocessing import StandardScaler

from Model.model_major import *
# from support_nested_lstm import NestedLSTM        ##### NLSTM

from Support.support_wavelet import *
from Support.support_VMD import VMD

from Part.part_evaluate import *
from Part.part_data_preprocessing import *

warnings.filterwarnings("ignore")

########################################################################


def load_data_ts(trainNum, testNum, startNum, data):
    print('General_data loading.')

    global ahead_num
    # all_data_checked = data

    targetData = data

    global x_mode

    time_series_y = create_time_series(targetData, ahead_num)

    allX = np.c_[time_series_y]

    allX = allX.T
    print("\nallX:", allX.shape)

    ###########======================================

    trainX = allX[:, : trainNum]
    trainY = targetData.T[:, ahead_num: trainNum + ahead_num]
    testX = allX[:, trainNum:]
    testY = targetData.T[:, trainNum + ahead_num: (trainNum + testNum)]

    # print("allX:", allX.shape)
    # print("trainX:", trainX.shape)
    # print("trainY:", trainY.shape)
    # print("testX:", testX.shape)
    # print("testY:", testY.shape)

    trainY = trainY.flatten()  # 降维
    testY = testY.flatten()  # 降维
    trainX = trainX.T
    testX = testX.T

    print('load_data complete.\n')

    return trainX, trainY, testX, testY


def load_data_wvlt(trainNum, testNum, startNum, data):
    print('wavelet_data loading.')

    global ahead_num
    # all_data_checked = data
    targetData = data

    testY = None
    global wvlt_lv

    wavefun = pywt.Wavelet('db1')
    coeffs = swt_decom(targetData, wavefun, wvlt_lv)

    ### 测试滤波效果
    wvlt_level_list = []
    for wvlt_level in range(len(coeffs)):
        wvlt_trainX, wvlt_trainY, wvlt_testX, wvlt_testY = create_data(coeffs[wvlt_level], trainNum, ahead_num)
        wvlt_level_part = [wvlt_trainX, wvlt_trainY, wvlt_testX, wvlt_testY]
        wvlt_level_list.append(wvlt_level_part)

    print('load_data complete.\n')

    return wvlt_level_list, testY


def load_data_general(start_num, interval_ori):
    Dict_loaded_data = {}

    global interval
    interval = interval_ori

    # training number
    startNum = start_num
    trainNum = (24 * 1000) // interval
    testNum = ((24 * 100) // interval) + ahead_num

    global num
    num = 12
    filename1 = "dataset\\PRSA_Data_"
    filename2 = ".csv"
    filename = [filename1, filename2]

    global x_mode
    x_mode = 2

    global wvlt_lv
    wvlt_lv = 3

    dataset_list = read_csv_all(filename, trainNum, testNum, startNum, num, interval)
    dataset_PM2 = dataset_list[0]  # PM2
    dataset_PM10 = dataset_list[1]  # PM10
    dataset_SO2 = dataset_list[2]  # SO2
    dataset_NO2 = dataset_list[3]  # NO2
    dataset_CO = dataset_list[4]  # CO
    dataset_O3 = dataset_list[5]  # O3
    # dataset = dataset_list[6]      #TEMP
    # dataset = dataset_list[7]      #PRES
    # dataset = dataset_list[8]      #DEWP

    # line_width = 1.5
    # plt.figure(figsize=(19, 5))
    # plt.plot((dataset_PM2 / (dataset_PM2.max() - dataset_PM2.min()))[0:200], label="PM2", linewidth=line_width)
    # plt.plot((dataset_PM10 / (dataset_PM10.max() - dataset_PM10.min()))[0:200], label="PM10", linewidth=line_width)
    # plt.plot((dataset_SO2 / (dataset_SO2.max() - dataset_SO2.min()))[0:200], label="SO2", linewidth=line_width)
    # plt.plot((dataset_NO2 / (dataset_NO2.max() - dataset_NO2.min()))[0:200], label="NO2", linewidth=line_width)
    # plt.plot((dataset_CO / (dataset_CO.max() - dataset_CO.min()))[0:200], label="CO", linewidth=line_width)
    # plt.plot((dataset_O3 / (dataset_O3.max() - dataset_O3.min()))[0:200], label="O3", linewidth=line_width)
    # plt.subplots_adjust(left=0.03, bottom=0.1, right=0.99, top=0.95)
    # plt.grid(True, linestyle=":", color="gray", linewidth="0.5", axis='both')
    # plt.xlabel("Time(Hours)")
    # plt.title('Multivariate trend Comparison')
    # plt.legend(loc='best')
    # plt.show()

    global scaler_PM2
    dataset_PM2 = np.array(dataset_PM2[startNum + 1: startNum + trainNum + testNum + 1]).reshape(-1, 1)
    scaler_PM2 = StandardScaler(copy=True, with_mean=True, with_std=True)
    dataset_PM2 = scaler_PM2.fit_transform(dataset_PM2)

    global scaler_PM10
    dataset_PM10 = np.array(dataset_PM10[startNum + 1: startNum + trainNum + testNum + 1]).reshape(-1, 1)
    scaler_PM10 = StandardScaler(copy=True, with_mean=True, with_std=True)
    dataset_PM10 = scaler_PM10.fit_transform(dataset_PM10)

    global scaler_SO2
    dataset_SO2 = np.array(dataset_SO2[startNum + 1: startNum + trainNum + testNum + 1]).reshape(-1, 1)
    scaler_SO2 = StandardScaler(copy=True, with_mean=True, with_std=True)
    dataset_SO2 = scaler_SO2.fit_transform(dataset_SO2)

    global scaler_NO2
    dataset_NO2 = np.array(dataset_NO2[startNum + 1: startNum + trainNum + testNum + 1]).reshape(-1, 1)
    scaler_NO2 = StandardScaler(copy=True, with_mean=True, with_std=True)
    dataset_NO2 = scaler_NO2.fit_transform(dataset_NO2)

    global scaler_CO
    dataset_CO = np.array(dataset_CO[startNum + 1: startNum + trainNum + testNum + 1]).reshape(-1, 1)
    scaler_CO = StandardScaler(copy=True, with_mean=True, with_std=True)
    dataset_CO = scaler_CO.fit_transform(dataset_CO)

    global scaler_O3
    dataset_O3 = np.array(dataset_O3[startNum + 1: startNum + trainNum + testNum + 1]).reshape(-1, 1)
    scaler_O3 = StandardScaler(copy=True, with_mean=True, with_std=True)
    dataset_O3 = scaler_O3.fit_transform(dataset_O3)

    # line_width = 1.5
    # plt.figure(figsize=(19, 5))
    # plt.plot(dataset_PM2[0:200, :], label="PM2", linewidth=line_width)
    # plt.plot(dataset_PM10[0:200, :], label="PM10", linewidth=line_width)
    # plt.plot(dataset_SO2[0:200, :], label="SO2", linewidth=line_width)
    # plt.plot(dataset_NO2[0:200, :], label="NO2", linewidth=line_width)
    # plt.plot(dataset_CO[0:200, :], label="CO", linewidth=line_width)
    # plt.plot(dataset_O3[0:200, :], label="O3", linewidth=line_width)
    # plt.subplots_adjust(left=0.03, bottom=0.1, right=0.99, top=0.95)
    # plt.grid(True, linestyle=":", color="gray", linewidth="0.5", axis='both')
    # plt.xlabel("Time(Hours)")
    # plt.title('Multivariate trend Comparison')
    # plt.legend(loc='best')
    # plt.show()

    x_train, y_train, x_test, y_test = load_data_ts(trainNum, testNum, startNum, dataset_PM2)
    x_train1, y_train1, x_test1, y_test1 = load_data_ts(trainNum, testNum, startNum, dataset_PM10)
    x_train2, y_train2, x_test2, y_test2 = load_data_ts(trainNum, testNum, startNum, dataset_SO2)
    x_train3, y_train3, x_test3, y_test3 = load_data_ts(trainNum, testNum, startNum, dataset_NO2)
    x_train4, y_train4, x_test4, y_test4 = load_data_ts(trainNum, testNum, startNum, dataset_CO)
    x_train5, y_train5, x_test5, y_test5 = load_data_ts(trainNum, testNum, startNum, dataset_O3)

    wvlt_list, _ = load_data_wvlt(trainNum, testNum, startNum, dataset_PM2)
    wvlt_list1, _ = load_data_wvlt(trainNum, testNum, startNum, dataset_PM10)
    wvlt_list2, _ = load_data_wvlt(trainNum, testNum, startNum, dataset_SO2)
    wvlt_list3, _ = load_data_wvlt(trainNum, testNum, startNum, dataset_NO2)
    wvlt_list4, _ = load_data_wvlt(trainNum, testNum, startNum, dataset_CO)
    wvlt_list5, _ = load_data_wvlt(trainNum, testNum, startNum, dataset_O3)

    train_x_list = [x_train.reshape(-1, ahead_num, 1),
                    x_train1.reshape(-1, ahead_num, 1),
                    x_train2.reshape(-1, ahead_num, 1),
                    x_train3.reshape(-1, ahead_num, 1),
                    x_train4.reshape(-1, ahead_num, 1),
                    x_train5.reshape(-1, ahead_num, 1)]

    test_x_list = [x_test.reshape(-1, ahead_num, 1),
                   x_test1.reshape(-1, ahead_num, 1),
                   x_test2.reshape(-1, ahead_num, 1),
                   x_test3.reshape(-1, ahead_num, 1),
                   x_test4.reshape(-1, ahead_num, 1),
                   x_test5.reshape(-1, ahead_num, 1)]

    train_y_list = [y_train, y_train1, y_train2, y_train3, y_train4, y_train5]
    test_y_list = [y_test, y_test1, y_test2, y_test3, y_test4]

    wvlt_trX_list = []
    wvlt_teX_list = []
    for i_wvlt in range(len(wvlt_list)):
        wvlt_trX = np.reshape(wvlt_list[i_wvlt][0],
                              (wvlt_list[i_wvlt][0].shape[0],
                               wvlt_list[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list[i_wvlt][2],
                              (wvlt_list[i_wvlt][2].shape[0],
                               wvlt_list[i_wvlt][2].shape[1], 1))
        wvlt_trX_list.append(wvlt_trX)
        wvlt_teX_list.append(wvlt_teX)

    wvlt_trX_list1 = []
    wvlt_teX_list1 = []
    for i_wvlt in range(len(wvlt_list1)):
        wvlt_trX = np.reshape(wvlt_list1[i_wvlt][0],
                              (wvlt_list1[i_wvlt][0].shape[0],
                               wvlt_list1[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list1[i_wvlt][2],
                              (wvlt_list1[i_wvlt][2].shape[0],
                               wvlt_list1[i_wvlt][2].shape[1], 1))
        wvlt_trX_list1.append(wvlt_trX)
        wvlt_teX_list1.append(wvlt_teX)

    wvlt_trX_list2 = []
    wvlt_teX_list2 = []
    for i_wvlt in range(len(wvlt_list2)):
        wvlt_trX = np.reshape(wvlt_list2[i_wvlt][0],
                              (wvlt_list2[i_wvlt][0].shape[0],
                               wvlt_list2[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list2[i_wvlt][2],
                              (wvlt_list2[i_wvlt][2].shape[0],
                               wvlt_list2[i_wvlt][2].shape[1], 1))
        wvlt_trX_list2.append(wvlt_trX)
        wvlt_teX_list2.append(wvlt_teX)

    wvlt_trX_list3 = []
    wvlt_teX_list3 = []
    for i_wvlt in range(len(wvlt_list3)):
        wvlt_trX = np.reshape(wvlt_list3[i_wvlt][0],
                              (wvlt_list3[i_wvlt][0].shape[0],
                               wvlt_list3[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list3[i_wvlt][2],
                              (wvlt_list3[i_wvlt][2].shape[0],
                               wvlt_list3[i_wvlt][2].shape[1], 1))
        wvlt_trX_list3.append(wvlt_trX)
        wvlt_teX_list3.append(wvlt_teX)

    wvlt_trX_list4 = []
    wvlt_teX_list4 = []
    for i_wvlt in range(len(wvlt_list4)):
        wvlt_trX = np.reshape(wvlt_list4[i_wvlt][0],
                              (wvlt_list4[i_wvlt][0].shape[0],
                               wvlt_list4[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list4[i_wvlt][2],
                              (wvlt_list4[i_wvlt][2].shape[0],
                               wvlt_list4[i_wvlt][2].shape[1], 1))
        wvlt_trX_list4.append(wvlt_trX)
        wvlt_teX_list4.append(wvlt_teX)

    wvlt_trX_list5 = []
    wvlt_teX_list5 = []
    for i_wvlt in range(len(wvlt_list5)):
        wvlt_trX = np.reshape(wvlt_list5[i_wvlt][0],
                              (wvlt_list5[i_wvlt][0].shape[0],
                               wvlt_list5[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list5[i_wvlt][2],
                              (wvlt_list5[i_wvlt][2].shape[0],
                               wvlt_list5[i_wvlt][2].shape[1], 1))
        wvlt_trX_list5.append(wvlt_trX)
        wvlt_teX_list5.append(wvlt_teX)

    wvlt_list_train_MV = wvlt_trX_list + \
                         wvlt_trX_list1 + \
                         wvlt_trX_list2 + \
                         wvlt_trX_list3 + \
                         wvlt_trX_list4 + \
                         wvlt_trX_list5
    wvlt_list_test_MV = wvlt_teX_list + \
                        wvlt_teX_list1 + \
                        wvlt_teX_list2 + \
                        wvlt_teX_list3 + \
                        wvlt_teX_list4 + \
                        wvlt_teX_list5

    # #####culculate Accuracy by rampflag
    dataY = scaler_PM2.inverse_transform(y_test)
    dataY1 = scaler_PM10.inverse_transform(y_test1)
    dataY2 = scaler_SO2.inverse_transform(y_test2)
    dataY3 = scaler_NO2.inverse_transform(y_test3)
    dataY4 = scaler_CO.inverse_transform(y_test4)
    dataY5 = scaler_O3.inverse_transform(y_test5)

    dataY_list = [dataY, dataY1, dataY2, dataY3, dataY4, dataY5]

    minLen = 0
    print('Accuracy Flag:', minLen)

    y_rampflag = deal_flag(dataY, minLen)
    y_rampflag1 = deal_flag(dataY1, minLen)
    y_rampflag2 = deal_flag(dataY2, minLen)
    y_rampflag3 = deal_flag(dataY3, minLen)
    y_rampflag4 = deal_flag(dataY4, minLen)
    y_rampflag5 = deal_flag(dataY5, minLen)

    y_rampflag_list = [y_rampflag, y_rampflag1, y_rampflag2, y_rampflag3, y_rampflag4, y_rampflag5]

    Dict_loaded_data['train_x_list'] = train_x_list
    Dict_loaded_data['test_x_list'] = test_x_list
    Dict_loaded_data['train_y_list'] = train_y_list
    Dict_loaded_data['test_y_list'] = test_y_list

    Dict_loaded_data['wvlt_list'] = wvlt_list
    Dict_loaded_data['wvlt_list1'] = wvlt_list1
    Dict_loaded_data['wvlt_list2'] = wvlt_list2
    Dict_loaded_data['wvlt_list3'] = wvlt_list3
    Dict_loaded_data['wvlt_list4'] = wvlt_list4
    Dict_loaded_data['wvlt_list5'] = wvlt_list5
    Dict_loaded_data['wvlt_list_tr'] = wvlt_list_train_MV
    Dict_loaded_data['wvlt_list_te'] = wvlt_list_test_MV

    Dict_loaded_data['dataY_list'] = dataY_list
    Dict_loaded_data['y_rampflag_list'] = y_rampflag_list

    Dict_loaded_data['scaler_PM2'] = scaler_PM2
    Dict_loaded_data['scaler_PM10'] = scaler_PM10
    Dict_loaded_data['scaler_SO2'] = scaler_SO2
    Dict_loaded_data['scaler_NO2'] = scaler_NO2
    Dict_loaded_data['scaler_CO'] = scaler_CO
    Dict_loaded_data['scaler_O3'] = scaler_O3

    np.save('saved\\Loaded_data_saved.npy', Dict_loaded_data)
    print('Data Loaded and Saved Successfully.')

    return Dict_loaded_data


def MT(train_x_list, test_list, train_y_list, y_rampflag_list, dataY_list):

    model_name = 'MT'
    print(model_name + ' Start.')

    model = buildNLSTM_MultiV_v2(ahead_num)

    time_callback = TimeHistory()
    history = model.fit(train_x_list, train_y_list, epochs=16, validation_split=0.05, verbose=1, callbacks=[time_callback])
    train_time = time_callback.totaltime / 6

    predict_list = model.predict(test_list)

    print(model_name + ' Complete.')

    predict_PM2 = predict_list[0]
    predict_PM2 = predict_PM2.reshape(-1, )

    global scaler_PM2
    predict_PM2 = scaler_PM2.inverse_transform(predict_PM2)
    flag_PM2 = deal_flag(predict_PM2, minLen)
    accuracy_PM2 = deal_accuracy(y_rampflag_list[0], flag_PM2)

    global eva_output, result_all
    model_name = 'MT_PM2'
    model_name_short = 'MT_PM2'
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY_list[0],
                                           predict_PM2,
                                           accuracy_PM2,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    predict_list[0] = predict_PM2

    ###########################################################

    predict_PM10 = predict_list[1]
    predict_PM10 = predict_PM10.reshape(-1, )

    global scaler_PM10
    predict_PM10 = scaler_PM10.inverse_transform(predict_PM10)
    flag_PM10 = deal_flag(predict_PM10, minLen)
    accuracy_PM10 = deal_accuracy(y_rampflag_list[1], flag_PM10)

    model_name = 'MT_PM10'
    model_name_short = 'MT_PM10'
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY_list[1],
                                           predict_PM10,
                                           accuracy_PM10,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    predict_list[1] = predict_PM10

    ###########################################################

    predict_SO2 = predict_list[2]
    predict_SO2 = predict_SO2.reshape(-1, )

    global scaler_SO2
    predict_SO2 = scaler_SO2.inverse_transform(predict_SO2)
    flag_SO2 = deal_flag(predict_SO2, minLen)
    accuracy_SO2 = deal_accuracy(y_rampflag_list[2], flag_SO2)

    model_name = 'MT_SO2'
    model_name_short = 'MT_SO2'
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY_list[2],
                                           predict_SO2,
                                           accuracy_SO2,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    predict_list[2] = predict_SO2

    ###########################################################

    predict_NO2 = predict_list[3]
    predict_NO2 = predict_NO2.reshape(-1, )

    global scaler_NO2
    predict_NO2 = scaler_NO2.inverse_transform(predict_NO2)
    flag_NO2 = deal_flag(predict_NO2, minLen)
    accuracy_NO2 = deal_accuracy(y_rampflag_list[3], flag_NO2)

    model_name = 'MT_NO2'
    model_name_short = 'MT_NO2'
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY_list[3],
                                           predict_NO2,
                                           accuracy_NO2,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    predict_list[3] = predict_NO2

    ###########################################################

    predict_CO = predict_list[4]
    predict_CO = predict_CO.reshape(-1, )

    global scaler_CO
    predict_CO = scaler_CO.inverse_transform(predict_CO)
    flag_CO = deal_flag(predict_CO, minLen)
    accuracy_CO = deal_accuracy(y_rampflag_list[4], flag_CO)

    model_name = 'MT_CO'
    model_name_short = 'MT_CO'
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY_list[4],
                                           predict_CO,
                                           accuracy_CO,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    predict_list[4] = predict_CO

    ###########################################################

    predict_O3 = predict_list[5]
    predict_O3 = predict_O3.reshape(-1, )

    global scaler_O3
    predict_O3 = scaler_O3.inverse_transform(predict_O3)
    flag_O3 = deal_flag(predict_O3, minLen)
    accuracy_O3 = deal_accuracy(y_rampflag_list[5], flag_O3)

    model_name = 'MT_O3'
    model_name_short = 'MT_O3'
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY_list[5],
                                           predict_O3,
                                           accuracy_O3,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    predict_list[5] = predict_O3

    return predict_list, history.history


def MTMC_wvlt(train_x_list, test_list, train_y_list, y_rampflag_list, dataY_list):

    model_name = 'MTMC'
    print(model_name + ' Start.')

    model = buildNLSTM_MultiV_v4(ahead_num)

    time_callback = TimeHistory()
    history = model.fit(train_x_list, train_y_list, epochs=16, validation_split=0.05, verbose=1, callbacks=[time_callback])
    train_time = time_callback.totaltime / 6

    predict_list = model.predict(test_list)

    print(model_name + ' Complete.')

    predict_PM2 = predict_list[0]
    predict_PM2 = predict_PM2.reshape(-1, )

    global scaler_PM2
    predict_PM2 = scaler_PM2.inverse_transform(predict_PM2)
    flag_PM2 = deal_flag(predict_PM2, minLen)
    accuracy_PM2 = deal_accuracy(y_rampflag_list[0], flag_PM2)

    global eva_output, result_all
    model_name = 'MTMC_PM2'
    model_name_short = 'MTMC_PM2'
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY_list[0],
                                           predict_PM2,
                                           accuracy_PM2,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    predict_list[0] = predict_PM2

    ###########################################################

    predict_PM10 = predict_list[1]
    predict_PM10 = predict_PM10.reshape(-1, )

    global scaler_PM10
    predict_PM10 = scaler_PM10.inverse_transform(predict_PM10)
    flag_PM10 = deal_flag(predict_PM10, minLen)
    accuracy_PM10 = deal_accuracy(y_rampflag_list[1], flag_PM10)

    model_name = 'MTMC_PM10'
    model_name_short = 'MTMC_PM10'
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY_list[1],
                                           predict_PM10,
                                           accuracy_PM10,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    predict_list[1] = predict_PM10

    ###########################################################

    predict_SO2 = predict_list[2]
    predict_SO2 = predict_SO2.reshape(-1, )

    global scaler_SO2
    predict_SO2 = scaler_SO2.inverse_transform(predict_SO2)
    flag_SO2 = deal_flag(predict_SO2, minLen)
    accuracy_SO2 = deal_accuracy(y_rampflag_list[2], flag_SO2)

    model_name = 'MTMC_SO2'
    model_name_short = 'MTMC_SO2'
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY_list[2],
                                           predict_SO2,
                                           accuracy_SO2,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    predict_list[2] = predict_SO2

    ###########################################################

    predict_NO2 = predict_list[3]
    predict_NO2 = predict_NO2.reshape(-1, )

    global scaler_NO2
    predict_NO2 = scaler_NO2.inverse_transform(predict_NO2)
    flag_NO2 = deal_flag(predict_NO2, minLen)
    accuracy_NO2 = deal_accuracy(y_rampflag_list[3], flag_NO2)

    model_name = 'MTMC_NO2'
    model_name_short = 'MTMC_NO2'
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY_list[3],
                                           predict_NO2,
                                           accuracy_NO2,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    predict_list[3] = predict_NO2

    ###########################################################

    predict_CO = predict_list[4]
    predict_CO = predict_CO.reshape(-1, )

    global scaler_CO
    predict_CO = scaler_CO.inverse_transform(predict_CO)
    flag_CO = deal_flag(predict_CO, minLen)
    accuracy_CO = deal_accuracy(y_rampflag_list[4], flag_CO)

    model_name = 'MTMC_CO'
    model_name_short = 'MTMC_CO'
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY_list[4],
                                           predict_CO,
                                           accuracy_CO,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    predict_list[4] = predict_CO

    ###########################################################

    predict_O3 = predict_list[5]
    predict_O3 = predict_O3.reshape(-1, )

    global scaler_O3
    predict_O3 = scaler_O3.inverse_transform(predict_O3)
    flag_O3 = deal_flag(predict_O3, minLen)
    accuracy_O3 = deal_accuracy(y_rampflag_list[5], flag_O3)

    model_name = 'MTMC_O3'
    model_name_short = 'MTMC_O3'
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY_list[5],
                                           predict_O3,
                                           accuracy_O3,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    predict_list[5] = predict_O3

    return predict_list, history.history

#########################################################################


def neo_prediction(start_num, interval_ori):

    # random_state = np.random.RandomState(7)
    np.random.RandomState(7)

    # lookback number
    global ahead_num
    ahead_num = 4

    global minLen
    minLen = 0

    Dict_load = load_data_general(start_num, interval_ori)

    #########################################################################################

    train_x_list = Dict_load['train_x_list']
    test_x_list = Dict_load['test_x_list']
    train_y_list = Dict_load['train_y_list']

    wvlt_list_train_MV = Dict_load['wvlt_list_tr']
    wvlt_list_test_MV = Dict_load['wvlt_list_te']

    dataY_list = Dict_load['dataY_list']
    y_rampflag_list = Dict_load['y_rampflag_list']

    global scaler_PM2, scaler_PM10, scaler_SO2, scaler_NO2, scaler_CO, scaler_O3
    scaler_PM2 = Dict_load['scaler_PM2']
    scaler_PM10 = Dict_load['scaler_PM10']
    scaler_SO2 = Dict_load['scaler_SO2']
    scaler_NO2 = Dict_load['scaler_NO2']
    scaler_CO = Dict_load['scaler_CO']
    scaler_O3 = Dict_load['scaler_O3']

    ######=========================Modelling and Predicting=========================#####
    print("======================================================")
    global eva_output, result_all
    eva_output = '\nEvaluation.'
    result_all = []

    predict_MTMC_wvlt, history_MTMC = \
        MTMC_wvlt(wvlt_list_train_MV,
                  wvlt_list_test_MV,
                  train_y_list,
                  y_rampflag_list,
                  dataY_list)

    predict_MT, history_MT = \
        MT(train_x_list,
           test_x_list,
           train_y_list,
           y_rampflag_list,
           dataY_list)

    Dict_Prediction = {}
    Dict_Prediction['real_data'] = dataY_list
    Dict_Prediction['predict_MTMC_wvlt'] = predict_MTMC_wvlt
    Dict_Prediction['predict_MT'] = predict_MT
    Dict_Prediction['history_MTMC'] = history_MTMC
    Dict_Prediction['history_MT'] = history_MT

    print(eva_output)

    # # print(result_all)
    save_file_name = "result\\result_MTMC.csv"

    csv_file = open(save_file_name, "w", newline="")  # 创建csv文件
    writer = csv.writer(csv_file)  # 创建写的对象
    writer.writerow(["", "MAE", "RMSE", "MAPE", "R2", "ACC", "TIME"])
    for wtr in range(len(result_all)):
        writer.writerow(result_all[wtr])

    # backup for Denied permission.
    save_file_name = "result\\result_MTMC_backup.csv"

    csv_file = open(save_file_name, "w", newline="")  # 创建csv文件
    writer = csv.writer(csv_file)  # 创建写的对象
    writer.writerow(["", "MAE", "RMSE", "MAPE", "R2", "ACC", "TIME"])
    for wtr in range(len(result_all)):
        writer.writerow(result_all[wtr])

    Dict_Prediction['result_all'] = result_all

    print('\nComplete.')

    print('The result shall be saved.')
    np.save('saved\\prediction_ts4_MTMC.npy', Dict_Prediction)
    print('The result is saved successfully.')

    return result_all


if __name__ == "__main__":

    start_num = 0
    interval_ori = 1
    _ = neo_prediction(start_num, interval_ori)

