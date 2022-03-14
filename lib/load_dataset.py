import os
import numpy as np


def load_st_dataset(dataset, input_dim, node_num):
    #output B, N, D (sequence_length, num_of_vertices, num_of_features)
    if dataset == 'PeMSD4':
        data_path = os.path.join('./data/PeMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
        # 合并时间
        # t_step = 6
        # n = int(data.shape[0] / t_step)
        # stop_flow_n = []
        # for i in range(data.shape[1]):
        #     temp = []
        #     for j in range(n):
        #         temp.append(sum(data[j*t_step:(j+1)*t_step, i]))
        #     stop_flow_n.append([temp])
        # data = np.array(stop_flow_n).transpose(2,0,1).reshape(n, -1)
    
    elif dataset == 'PeMSD8':
        data_path = os.path.join('./data/PeMSD8/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'bus':
        if input_dim == 1:
            data_path = os.path.join('/home/chenxin/first_year/bus/constrast_experiment/DCRNN/data_processing/data/bus/stop_flow_15_{}.npy'.format(node_num))
            stop_flow = np.load(data_path).astype(np.float64)
            data = stop_flow
            data = stop_flow.transpose((2,0,1))
        else:
            data_path = os.path.join('/content/drive/MyDrive/Colab Notebooks/bus/bus_data/clean_data/flow_graph_all/stop_flow_15_{}.npy'.format(node_num))
            stop_flow = np.load(data_path).astype(np.float64)
            data = stop_flow
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data