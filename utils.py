import torch
import numpy as np
import torch.utils.data as Data
import random
from sklearn.metrics import accuracy_score,precision_recall_fscore_support, top_k_accuracy_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#训练集、验证集、测试集的划分
train_percent = 0.8
val_percent = 0.1
test_percent = 0.1

#屏幕padding，每个屏幕内控件数不同，需要进行padding
def screen_padding(screen , size):
    """
    screen: 屏幕的tensor
    size: padding的目标size即设置的屏幕内最大控件数
    """
    sc_mask = torch.ones(size) == 0
    ZeroPad = torch.nn.ZeroPad2d(padding=(0, 0, 0, size - screen.shape[0]))
    pad_s = ZeroPad(screen)
    for i in range(screen.shape[0] , size):
        sc_mask[i] = True
    return pad_s , sc_mask

#由于数据量大，无法单次加载全部数据，重写了dataset的部分函数
class MyDataset(Data.Dataset):
    def __init__(self , file_path , nraws , sc_size , winsize , tag , shuffle=False):
        """
        file_path: the path to the dataset file
        nraws: each time put nraws sample into memory for shuffle
        shuffle: whether the data need to shuffle
        """
        # apppath = file_path + 'app.txt'
        self.locpath = file_path + 'loclist.npy'
        self.scpath = file_path + 'screenlist.npy'
        self.upperpath = file_path + 'upperlist.npy'
        self.contextpath = file_path + 'context.pt'
        self.lenpath = file_path + 'lenlist.npy'
        
        self.sc_size = sc_size
        self.nraws = nraws
        self.winsize = winsize
        self.shuffle = shuffle
        self.begin = 0
        self.tag = tag
        loclist=np.load(self.locpath, allow_pickle=True)
        file_raws = loclist.size 
        self.total_raws = file_raws - winsize + 1
        if self.tag == 'train':
            self.file_raws = int(self.total_raws * 0.8)
        if self.tag == 'val':
            self.file_raws = int(self.total_raws * 0.1)
        if self.tag == 'test':
            self.file_raws = int(self.total_raws * 0.1)
            
 
    def initial(self):
        if self.tag == 'train':
            self.begin = 0
        if self.tag == 'val':
            self.begin = int(self.total_raws * 0.8)
        if self.tag == 'test':
            self.begin = int(self.total_raws * 0.9)
        # self.begin = 0
        self.loclist = torch.tensor(np.load(self.locpath, allow_pickle=True)).unsqueeze(1)
        self.screenlist = np.load(self.scpath, allow_pickle=True)
        self.upperlist = np.load(self.upperpath, allow_pickle=True)
        self.contextlist = torch.load(self.contextpath)
        self.lenlist = torch.tensor(np.load(self.lenpath, allow_pickle=True)).unsqueeze(1)

        self.samples = list()
        
        # put nraw samples into memory
        for i in range(self.nraws - self.winsize):
            new_screenlist = []
            new_upperlist = []
            sc_mask_list = []
            new_loclist = []
            new_contextlist = []
            new_lenlist = []
            
            if self.begin + i + 1> self.total_raws:
                break
            for s in range(self.begin + i , self.begin + i + self.winsize):
                new_s , mask = screen_padding(self.screenlist[s] , self.sc_size)
                new_upper = self.upperlist[s]
                
                new_screenlist.append(new_s.unsqueeze(0))
                new_upperlist.append(new_upper.unsqueeze(0))
                
                sc_mask_list.append(mask.unsqueeze(0)) 
                new_loclist.append(self.loclist[s])
                new_contextlist.append(self.contextlist[:,s].unsqueeze(0))
                new_lenlist.append(self.lenlist[s])
                
            t_screenlist = torch.cat(new_screenlist)
            t_upperlist = torch.cat(new_upperlist)
            t_mask_list = torch.cat(sc_mask_list)
            t_loclist = torch.cat(new_loclist)
            t_contextlist = torch.cat(new_contextlist)
            t_lenlist = torch.cat(new_lenlist)
            
            
            data = (t_screenlist , t_mask_list , t_loclist , t_contextlist , t_upperlist, t_lenlist)   # data contains the feature and label
            if not data == None:
                self.samples.append(data)
            else:
                break
        self.begin = self.begin + self.nraws - self.winsize   
        self.current_sample_num = len(self.samples)
        self.index = list(range(self.current_sample_num))
        
        if self.shuffle:
            random.shuffle(self.samples)
 
    def __len__(self):
        return self.file_raws
 
    def __getitem__(self,item):
        idx = self.index[0]
        rdata = self.samples[idx]
        del(self.index[0])
        if len(self.index)  ==  0:
            self.samples.clear()
            
            for i in range(self.nraws - self.winsize):
                new_screenlist = []
                new_upperlist = []
                
                sc_mask_list = []
                new_loclist = []
                new_contextlist = []
                new_lenlist = []

                if self.begin + i + 1> self.total_raws:
                    break
                for s in range(self.begin + i , self.begin + i + self.winsize):
                    new_s , mask = screen_padding(self.screenlist[s] , self.sc_size)
                    new_upper = self.upperlist[s]

                    new_screenlist.append(new_s.unsqueeze(0))
                    new_upperlist.append(new_upper.unsqueeze(0))

                    sc_mask_list.append(mask.unsqueeze(0)) 
                    new_loclist.append(self.loclist[s])
                    new_contextlist.append(self.contextlist[:,s].unsqueeze(0))
                    new_lenlist.append(self.lenlist[s])
                t_screenlist = torch.cat(new_screenlist)
                t_upperlist = torch.cat(new_upperlist)
                t_mask_list = torch.cat(sc_mask_list)
                t_loclist = torch.cat(new_loclist)
                t_contextlist = torch.cat(new_contextlist)
                t_lenlist = torch.cat(new_lenlist)


                data = (t_screenlist , t_mask_list , t_loclist , t_contextlist , t_upperlist, t_lenlist)   # data contains the feature and label
                if not data == None:
                    self.samples.append(data)
                else:
                    break  
            
            self.begin = self.begin + self.nraws - self.winsize       
            self.current_sample_num = len(self.samples)
            self.index = list(range(self.current_sample_num))
            if self.shuffle:
                random.shuffle(self.samples) 
        return rdata
    

def evaluation(output, label, len_list, pattern='micro'):
    l_label = label.tolist()
    l_lenlist = len_list.tolist()
    y_pred_pre1 = []
    y_pred_pre3 = []
    y_pred_pre5 = []

    for i in range(len(l_label)):
        res = torch.argsort(output, 1, descending=True)[i,:].tolist()
        indice1 = []
        indice3 = []
        indice5 = []
        bound = l_lenlist[i]
        for r in res:
            if r <= bound:
                if len(indice1) < 1:
                    indice1.append(r)
                if len(indice3) < 3:
                    indice3.append(r)
                if len(indice5) < 5:
                    indice5.append(r)
            if len(indice5) == 5:
                break
        if bound > 80:
            y_pred_pre1.append(l_label[i])
            y_pred_pre3.append(l_label[i])
            y_pred_pre5.append(l_label[i])
        else:
            if l_label[i] in indice1:
                y_pred_pre1.append(l_label[i])
            else:
                y_pred_pre1.append(indice1[0])
            if l_label[i] in indice3:
                y_pred_pre3.append(l_label[i])
            else:
                y_pred_pre3.append(indice3[0])
            if l_label[i] in indice5:
                y_pred_pre5.append(l_label[i])
            else:
                y_pred_pre5.append(indice5[0])
    accuracy1 = accuracy_score(l_label, y_pred_pre1)
    precision1, recall1, F11, _ = precision_recall_fscore_support(l_label, y_pred_pre1, average=pattern)
    accuracy3 = accuracy_score(l_label, y_pred_pre3)
    precision3, recall3, F13, _ = precision_recall_fscore_support(l_label, y_pred_pre3, average=pattern)
    accuracy5 = accuracy_score(l_label, y_pred_pre5)
    precision5, recall5, F15, _ = precision_recall_fscore_support(l_label, y_pred_pre5, average=pattern) 
    return accuracy1, precision1, recall1, F13,accuracy3, precision3, recall3, F13,accuracy5, precision5, recall5, F15

def eval_handle(val_inputs, model):
    out_list = []
    label_list = []
    len_list = []
    tmp = []
    s = 0
    for step, (t_screenlist , t_mask_list , t_loclist , t_contextlist, t_upperlist, t_lenlist) in enumerate(val_inputs):
        tmp = t_screenlist
        s = step + 1
        in_datas = t_screenlist.to(device)
        in_label = t_loclist.to(device)
        in_upper = t_upperlist.to(device)
        mask_list = t_mask_list.to(device)
        context_list = t_contextlist.to(device)
        in_len = t_lenlist.to(device)
        out = model(in_datas, in_label , mask_list , context_list , in_upper)

        out_list.append(out.squeeze(2))
        label_list.append(in_label[:,-1])
        len_list.append(in_len[:,-1])
        
    output = torch.cat(tuple(out_list)).reshape(s, tmp.shape[2])
    label =  torch.cat(tuple(label_list))
    lenlist =  torch.cat(tuple(len_list))
    
    accuracy1, precision1, recall1, F11 , accuracy3, precision3, recall3, F13,accuracy5, precision5, recall5, F15 = evaluation(output, label , lenlist)
    return accuracy1, precision1, recall1, F11 , accuracy3, precision3, recall3, F13,accuracy5, precision5, recall5, F15