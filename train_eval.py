import time
import os
import torch
import numpy as np
from torch import nn
import torch.utils.data as Data
from sklearn.metrics import accuracy_score,precision_recall_fscore_support, top_k_accuracy_score
from layers import ESPACE
from utils import MyDataset
from utils import eval_handle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 1

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
# 数据路径
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
loss_list = []
loss_fuction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
state = {}
start_time = time.time()
best_x, best_epoch = 0, 0
pass_epoch = 0

for epoch in range(pass_epoch , pass_epoch + num_epochs):
    start_time = time.time()
    total_step = 0
    train_loss = 0
    for subpath in pathlist:
        datapath = inputPath + subpath
        train_dataset = MyDataset(datapath , nraws , sc_size , win_size + 1 , 'train' , shuffle=False)
        train_dataset.initial()
        dataloader = Data.DataLoader(dataset = train_dataset, batch_size = batch_size)
        model.train()

        for step, (t_screenlist , t_mask_list , t_loclist , t_contextlist , t_upperlist, t_lenlist) in enumerate(dataloader):
            total_step = total_step + 1 
            in_datas = t_screenlist.to(device)
            in_upper = t_upperlist.to(device)
            in_label = t_loclist.to(device)
            mask_list = t_mask_list.to(device)
            context_list = t_contextlist.to(device)
            optimizer.zero_grad()
            out = model(in_datas, in_label , mask_list , context_list , in_upper)
            loss = loss_fuction(out.squeeze(2), in_label[:,-1])
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
    loss_list.append(train_loss / total_step)
    save_loss=np.array(loss_list)
    # np.save('model/ESPACE_trainloss.npy', save_loss)
    print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, pass_epoch + num_epochs, train_loss / total_step))
    # val evaluate
    with torch.no_grad():
        laccuracy1, lprecision1, lrecall1, lF11 , laccuracy3, lprecision3, lrecall3, lF13 , laccuracy5, lprecision5, lrecall5, lF15 = [],[],[],[],[],[],[],[],[],[],[],[]
        for subpath in pathlist:
            datapath = inputPath + subpath
            model.eval()
            val_dataset = MyDataset(datapath , nraws , sc_size , win_size + 1 , 'val' , shuffle=False)
            val_dataset.initial()
            val_dataloader = Data.DataLoader(dataset = val_dataset)
            taccuracy1, tprecision1, trecall1, tF11 , taccuracy3, tprecision3, trecall3, tF13,taccuracy5, tprecision5, trecall5, tF15  = eval_handle(val_dataloader, model)

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

        print("Accuracy5: %.2f%%" % (accuracy5 * 100.0))
        print("Accuracy3: %.2f%%" % (accuracy3 * 100.0))
        print("Accuracy1: %.2f%%" % (accuracy1 * 100.0))
        if best_x <= accuracy5:
            best_x = accuracy5
            best_epoch = epoch
            # 保存模型
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': best_epoch}
            dir = "model/ESPACE_bestmodel.pt"
            torch.save(state, dir)
    end_time = time.time()
    print('time_cost:', end_time - start_time)
    print('best_epoch:', best_epoch)

    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': best_epoch}
    dir = "model/ESPACE_model.pt"
    torch.save(state, dir)

#模型评估
model.eval()
model_data = torch.load("model/ESPACE_bestmodel.pt")
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