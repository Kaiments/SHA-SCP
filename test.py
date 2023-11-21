import os
import torch
import numpy as np
import torch.utils.data as Data
from layers import ESPACE
from utils import MyDataset
from sklearn.metrics import accuracy_score,precision_recall_fscore_support, top_k_accuracy_score
from utils import eval_handle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper params
win_size = 1
sc_size = 100
batch_size = 512
nraws = 400
dec_seq_len_sc = sc_size
d_model_sc = 32
nhead_sc = 1
input_dim_seq = 64
dec_seq_len_seq = 10
out_seq_len_seq = 64
d_model_seq = 64
nhead_seq = 1
dim_feedforward_seq = 128
num_i = 128
num_h = 128
model_data = torch.load("model/ESPACE_bestmodel.pt")
inputPath = "prepared_data/"
files = os.listdir(inputPath)

pathlist = []
for file in files:
    Local = os.path.join(inputPath, file)
    if 'loclist' in Local:
        pathlist.append(file[:-11])

model = ESPACE(dec_seq_len_sc=dec_seq_len_sc,
               dec_seq_len_seq=win_size,
               d_model_sc=d_model_sc,
               nhead_sc=nhead_sc,
               input_dim_seq=input_dim_seq,
               out_seq_len_seq=out_seq_len_seq,
               d_model_seq=d_model_seq,
               nhead_seq=nhead_seq,
               dim_feedforward_seq=dim_feedforward_seq,
               num_i=num_i,
               num_h=num_h
               ).to(device)
#模型评估
model.eval()
#加载要测试的模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
model.load_state_dict(model_data['model'])
optimizer.load_state_dict(model_data['optimizer'])

with torch.no_grad():
    laccuracy1, lprecision1, lrecall1, lF11, laccuracy3, lprecision3, lrecall3, lF13, laccuracy5, lprecision5, lrecall5, lF15 = [], [], [], [], [], [], [], [], [], [], [], []
    for subpath in pathlist:
        datapath = inputPath + subpath
        test_dataset = MyDataset(datapath, nraws, sc_size, win_size + 1, 'test', shuffle=False)
        test_dataset.initial()
        test_dataloader = Data.DataLoader(dataset=test_dataset)

        taccuracy1, tprecision1, trecall1, tF11, taccuracy3, tprecision3, trecall3, tF13, taccuracy5, tprecision5, trecall5, tF15 = eval_handle(
            test_dataloader, model)

        laccuracy1.append(taccuracy1)
        lprecision1.append(tprecision1)
        lrecall1.append(trecall1)
        lF11.append(tF11)
        laccuracy3.append(taccuracy3)
        lprecision3.append(tprecision3)
        lrecall3.append(trecall3)
        lF13.append(tF13)
        laccuracy5.append(taccuracy5)
        lprecision5.append(tprecision5)
        lrecall5.append(trecall5)
        lF15.append(tF15)

    accuracy1 = np.mean(laccuracy1)
    precision1 = np.mean(lprecision1)
    recall1 = np.mean(lrecall1)
    F11 = np.mean(lF11)
    accuracy3 = np.mean(laccuracy3)
    precision3 = np.mean(lprecision3)
    recall3 = np.mean(lrecall3)
    F13 = np.mean(lF13)
    accuracy5 = np.mean(laccuracy5)
    precision5 = np.mean(lprecision5)
    recall5 = np.mean(lrecall5)
    F15 = np.mean(lF15)

    print("TestAccuracy5: %.2f%%" % (accuracy5 * 100.0))
    print("TestAccuracy3: %.2f%%" % (accuracy3 * 100.0))
    print("TestAccuracy1: %.2f%%" % (accuracy1 * 100.0))
