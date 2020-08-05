import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt


#####################################################################

def evaluate(variate_name, AQI_index, Dict_list, start_num, end_num):

    #################################################################

    predict_decideTree = Dict_list[AQI_index]['predict_decideTree']
    predict_randomForest = Dict_list[AQI_index]['predict_randomForest']
    predict_svr = Dict_list[AQI_index]['predict_svr']
    predict_mlp = Dict_list[AQI_index]['predict_mlp']

    predict_lstm = Dict_list[AQI_index]['predict_lstm']
    predict_lstm_emd = Dict_list[AQI_index]['predict_lstm_emd']
    predict_wavelet = Dict_list[AQI_index]['predict_wavelet']
    predict_VMD = Dict_list[AQI_index]['predict_VMD']

    predict_nlstm = Dict_list[AQI_index]['predict_nlstm']
    predict_EN = Dict_list[AQI_index]['predict_EN']
    predict_WN = Dict_list[AQI_index]['predict_WN']
    predict_VN = Dict_list[AQI_index]['predict_VN']

    predict_slstm = Dict_list[AQI_index]['predict_slstm']
    predict_ES = Dict_list[AQI_index]['predict_ES']
    predict_WS = Dict_list[AQI_index]['predict_WS']
    predict_VS = Dict_list[AQI_index]['predict_VS']

    predict_MTMC = Dict_list[-1]['predict_MTMC_wvlt'][AQI_index]
    predict_MT = Dict_list[-1]['predict_MT'][AQI_index]
    dataY = Dict_list[-1]['real_data'][AQI_index]

    result_all = Dict_list[AQI_index]['result_all']
    result_all_MTMC = Dict_list[-1]['result_all']
    result_all.append(result_all_MTMC[AQI_index])
    result_all.append(result_all_MTMC[AQI_index + 6])

    #################################################################

    predict_decideTree = predict_decideTree[start_num: start_num + end_num, ]
    predict_randomForest = predict_randomForest[start_num: start_num + end_num, ]
    predict_svr = predict_svr[start_num: start_num + end_num, ]
    predict_mlp = predict_mlp[start_num: start_num + end_num, ]

    predict_lstm = predict_lstm[start_num: start_num + end_num, ]
    predict_lstm_emd = predict_lstm_emd[start_num: start_num + end_num, ]
    predict_wavelet = predict_wavelet[start_num: start_num + end_num, ]
    predict_VMD = predict_VMD[start_num: start_num + end_num, ]

    predict_nlstm = predict_nlstm[start_num: start_num + end_num, ]
    predict_EN = predict_EN[start_num: start_num + end_num, ]
    predict_WN = predict_WN[start_num: start_num + end_num, ]
    predict_VN = predict_VN[start_num: start_num + end_num, ]

    predict_slstm = predict_slstm[start_num: start_num + end_num, ]
    predict_ES = predict_ES[start_num: start_num + end_num, ]
    predict_WS = predict_WS[start_num: start_num + end_num, ]
    predict_VS = predict_VS[start_num: start_num + end_num, ]

    predict_MTMC = predict_MTMC[start_num: start_num + end_num, ]
    predict_MT = predict_MT[start_num: start_num + end_num, ]
    dataY = dataY[start_num: start_num + end_num, ]

    #################################################################

    main_linewidth = 3
    second_linewidth = 1.75
    third_linewidth = 1.25

    plt.figure(1, figsize=(19, 5))
    plt.plot(dataY, "black", label="Real data", linewidth=main_linewidth, linestyle='--', marker='.')

    plt.plot(predict_decideTree, "aqua", label="Decision Tree", linewidth=third_linewidth)
    plt.plot(predict_randomForest, "lightsteelblue", label="Random Foreest", linewidth=third_linewidth)
    plt.plot(predict_svr, "cornflowerblue", label="SVR", linewidth=third_linewidth)
    plt.plot(predict_mlp, "royalblue", label="MLP", linewidth=third_linewidth)

    plt.plot(predict_lstm, "khaki", label="LSTM", linewidth=third_linewidth)
    plt.plot(predict_slstm, "orange", label="SLSTM", linewidth=third_linewidth)
    plt.plot(predict_nlstm, "darkorange", label="NLSTM", linewidth=third_linewidth)

    plt.plot(predict_lstm_emd, "limegreen", label="EMD-LSTM", linewidth=third_linewidth)
    plt.plot(predict_ES, "lightgreen", label="EMD-SLSTM", linewidth=third_linewidth)
    plt.plot(predict_EN, "springgreen", label="EMD-NLSTM", linewidth=third_linewidth)

    plt.plot(predict_wavelet, "thistle", label="Wavelet-LSTM", linewidth=third_linewidth)
    plt.plot(predict_WS, "violet", label="Wavelet-SLSTM", linewidth=third_linewidth)
    plt.plot(predict_WN, "fuchsia", label="Wavelet-NLSTM", linewidth=third_linewidth)

    plt.plot(predict_VMD, "gray", label="VMD-LSTM", linewidth=third_linewidth)
    plt.plot(predict_VS, "silver", label="VMD-SLSTM", linewidth=third_linewidth)
    plt.plot(predict_VN, "teal", label="VMD-NLSTM", linewidth=third_linewidth)

    plt.plot(predict_VN, "darkred", label="Multi-task", linewidth=third_linewidth)
    plt.plot(predict_MTMC, "red", label="proposed", linewidth=second_linewidth, marker='o')

    plt.subplots_adjust(left=0.04, bottom=0.1, right=0.99, top=0.95)
    plt.xlabel("Time(Hours)")
    plt.ylabel(str(variate_name) + " Concentration(ug/m^3)")
    plt.title('Forcasts and Actual Comparison of ' + str(variate_name))

    plt.grid(True, linestyle=":", color="gray", linewidth="0.5", axis='both')
    # plt.legend(loc='center right')
    plt.show()

    #############################################################

    save_file_name = "saved\\saved_result_" + str(variate_name) + ".csv"
    csv_file = open(save_file_name, "w", newline="")  # 创建csv文件
    writer = csv.writer(csv_file)
    writer.writerow(["", "MAE", "RMSE", "MAPE", "R2", "ACC", "TIME", str(variate_name)])
    for wtr in range(len(result_all)):
        writer.writerow(result_all[wtr])

    return None

#############################################################

Dict_MTMC = np.load('saved\\prediction_ts4_MTMC.npy', allow_pickle=True).item()
Dict_other0 = np.load('saved\\prediction_ts4_otherbase_PM2.npy', allow_pickle=True).item()
Dict_other1 = np.load('saved\\prediction_ts4_otherbase_PM10.npy', allow_pickle=True).item()
Dict_other2 = np.load('saved\\prediction_ts4_otherbase_SO2.npy', allow_pickle=True).item()
Dict_other3 = np.load('saved\\prediction_ts4_otherbase_NO2.npy', allow_pickle=True).item()
Dict_other4 = np.load('saved\\prediction_ts4_otherbase_CO.npy', allow_pickle=True).item()
Dict_other5 = np.load('saved\\prediction_ts4_otherbase_O3.npy', allow_pickle=True).item()

Dict_all = [Dict_other0, Dict_other1, Dict_other2, Dict_other3, Dict_other4, Dict_other5, Dict_MTMC]

evaluate('PM2.5', 0, Dict_all, 910, 72)
evaluate('PM10', 1, Dict_all, 910, 72)
evaluate('SO2', 2, Dict_all, 610, 72)
evaluate('NO2', 3, Dict_all, 1200, 72)
evaluate('CO', 4, Dict_all, 900, 72)
evaluate('O3', 5, Dict_all, 600, 72)

print('Load Complete.')