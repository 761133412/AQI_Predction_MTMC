
import csv
from Part.part_data_preprocessing import *

########################################################################

def main(start_num, interval_ori):
    # random_state = np.random.RandomState(7)
    np.random.RandomState(7)

    # lookback number
    global ahead_num
    ahead_num = 4

    global interval
    interval = interval_ori

    # training number
    startNum = start_num
    trainNum = (24 * 1000) // interval
    testNum = (24 * 20) // interval

    global num
    num = 12
    filename1 = "dataset\\PRSA_Data_"
    filename2 = ".csv"
    filename = [filename1, filename2]

    dataset_list = read_csv_all(filename, trainNum, testNum, startNum, num, interval)
    dataset_PM2 = dataset_list[0]  # PM2
    dataset_PM10 = dataset_list[1]  # PM10
    dataset_SO2 = dataset_list[2]  # SO2
    dataset_NO2 = dataset_list[3]  # NO2
    dataset_CO = dataset_list[4]  # CO
    dataset_O3 = dataset_list[5]  # O3

    multi_list = []
    for i in range(6):
        uni_list = []
        for j in range(6):
            PCC = kpr(dataset_list[i], dataset_list[j])
            uni_list.append(PCC)
        multi_list.append(uni_list)

    save_file_name = "../result/PCC.csv"
    csv_file = open(save_file_name, "w", newline="")  # 创建csv文件
    writer = csv.writer(csv_file)  # 创建写的对象
    for wtr in range(len(multi_list)):
        writer.writerow(multi_list[wtr])

    return None


if __name__ == "__main__":
    # start_num = 7000 * 24
    start_num = 2 * 1000 * 24
    interval_ori = 1
    _ = main(start_num, interval_ori)

