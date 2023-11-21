import torch
from torch import nn
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#transformer的位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x

class TransformerSeq(nn.Module):
    def __init__(self,
                 input_dim,
                 dec_seq_len,
                 out_seq_len,
                 d_model=516,
                 nhead=1,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=128,
                 dropout=0.1,
                 activation='relu',
                 custom_encoder=None,
                 custom_decoder=None):
        super(TransformerSeq, self).__init__()
        self.transform = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder,
        )
        self.pos = PositionalEncoding(d_model)
        self.enc_input_fc = nn.Linear(input_dim, d_model)
        self.dec_input_fc = nn.Linear(input_dim, d_model)
        self.out_fc = nn.Linear(dec_seq_len * d_model, out_seq_len)
        self.dec_seq_len = dec_seq_len

    def forward(self, x):
        x = x.transpose(0, 1)
        # embedding
        embed_encoder_input = self.enc_input_fc(x)
        embed_decoder_input = self.dec_input_fc(x[-self.dec_seq_len:, :])
        x = self.transform(embed_encoder_input, embed_decoder_input)
        # output
        x = x.transpose(0, 1)
        x = self.out_fc(x.flatten(start_dim=1))
        return x

#MLP
class MLP(torch.nn.Module):
    def __init__(self,
                 num_i,
                 num_h,
                 num_o):
        """
        num_i: 输入维度
        num_h: 隐层维度
        num_o: 输出维度
        """
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_o)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

#完整Google模型
class ESPACE(torch.nn.Module):
    def __init__(self,
                d_model_sc=32,
                dec_seq_len_sc = 100,
                nhead_sc=2,
                input_dim_seq=64,
                dec_seq_len_seq=10,
                out_seq_len_seq=64,
                d_model_seq=64,
                nhead_seq=1,
                num_encoder_layers_seq=6,
                num_decoder_layers_seq=6,
                dim_feedforward_seq=128,
                dropout_seq=0.1,
                activation_seq='relu',
                num_i=128,
                num_h=128,
                num_o=1,
                nlayers = 1):
        super(ESPACE,self).__init__()     
        
        self.demitionLayer1 = torch.nn.Linear(512,16)#改变原始维度
        self.demitionLayer2 = torch.nn.Linear(23,16)
        self.demitionLayer3 = torch.nn.Linear(2,32)
        self.demitionLayer_day = torch.nn.Linear(7,4)
        self.demitionLayer_hour = torch.nn.Linear(24,4)
        self.demitionLayer_interval = torch.nn.Linear(1,64)
        self.demitionLayer_app = torch.nn.Linear(1,8)
        
        self.updemitionLayer1 = torch.nn.Linear(512,8)   
        self.updemitionLayer2 = torch.nn.Linear(23,8)
        self.updemitionLayer3 = torch.nn.Linear(2,16)

        encoder_layers = nn.TransformerEncoderLayer(dim_feedforward=128,d_model=d_model_sc, nhead=nhead_sc)
        self.transformer_sc = nn.TransformerEncoder(encoder_layers, nlayers)
        
        upper_encoder_layers = nn.TransformerEncoderLayer(dim_feedforward=128,d_model=16, nhead=nhead_sc)
        self.transformer_upper_sc = nn.TransformerEncoder(upper_encoder_layers, nlayers)
        self.pos = PositionalEncoding(d_model_sc)
        self.transformer_seq=TransformerSeq(input_dim_seq,
                                  dec_seq_len_seq,
                                  out_seq_len_seq,
                                  d_model=d_model_seq,
                                  nhead=nhead_seq,
                                  num_encoder_layers=num_encoder_layers_seq,
                                  num_decoder_layers=num_decoder_layers_seq,
                                  dim_feedforward=dim_feedforward_seq,
                                  dropout=dropout_seq,
                                  activation=activation_seq,
                                  custom_encoder=None,
                                  custom_decoder=None)
        self.mlp=MLP(num_i=num_i,
                num_h=num_h,
                num_o=num_o)
        self.d_model_seq=d_model_seq
        self.dec_seq_len_seq = dec_seq_len_seq
        
    def forward(self, sclist , loclist , masklist , contextlist , upperlist):
    # upper维度16，sc维度32，context
        seq_input_list = []
        for idx in range(sclist.shape[1] - 1):#sclist:(batch , screen , objs , dim)
            upper_sc_text = self.updemitionLayer1(upperlist[:,idx,:,:512])
            upper_sc_type = self.updemitionLayer2(upperlist[:,idx,:,512:535])
            upper_sc_loc = self.updemitionLayer3(upperlist[:,idx,:,535:537])
            upper_sc = torch.concat((upper_sc_text,upper_sc_type),2)+upper_sc_loc
            upper_input_perbatch = self.transformer_upper_sc(upper_sc.transpose(1,0)) #(9 , batch , dim) 
            
            sc_text = self.demitionLayer1(sclist[:,idx,:,:512])
            sc_type = self.demitionLayer2(sclist[:,idx,:,512:535])
            sc_loc = self.demitionLayer3(sclist[:,idx,:,537:539])
            sc=torch.concat((sc_text,sc_type),2)+sc_loc
            
            context_w= self.demitionLayer_day(contextlist[:,idx,:7])
            context_r= self.demitionLayer_hour(contextlist[:,idx,7:31])
            context_i= self.demitionLayer_interval(contextlist[:,idx,31:32])
            
            context_app= self.demitionLayer_app(contextlist[:,idx,32:33])
            new_contextlist=torch.concat((context_w,context_r,context_app),1)
            
            sc = self.pos(sc)
            seq_input_perbatch = self.transformer_sc(sc.transpose(1,0)) #(objs , batch , dim)  
            seq_input_tl_tmp = []
            
            for ibatch in range(seq_input_perbatch.shape[1]):
                target = sclist[ibatch ,idx,int(loclist[ibatch,idx].item()),-1]
                upper_and_seq = torch.cat([upper_input_perbatch[:,ibatch,:][int(target.item())],seq_input_perbatch[:,ibatch,:][int(loclist[ibatch,idx].item())]])
                seq_input_tl_tmp.append(torch.cat([upper_and_seq ,new_contextlist[ibatch,:]]))#(dim + cxtdim)
            seq_input_list.append(torch.stack(tuple(seq_input_tl_tmp)))
        seq_input = torch.stack(tuple(seq_input_list)) + context_i#(seqlen , batch , newdim)
        
        next_target = []
        for iobj in range(sclist[:,-1,:,:].shape[1]):
            next_target.append(sclist[:,-1,iobj,-1])
            
        upper_next_text = self.updemitionLayer1(upperlist[:,-1,:,:512])
        upper_next_type = self.updemitionLayer2(upperlist[:,-1,:,512:535])
        upper_next_loc = self.updemitionLayer3(upperlist[:,-1,:,535:537])
        upper_next = torch.concat((upper_next_text,upper_next_type),2)+upper_next_loc
        upper_tmp_next = self.transformer_upper_sc(upper_next.transpose(1,0))#(9 , batch , dim)
        
        nextsc_text = self.demitionLayer1(sclist[:,-1,:,:512])
        nextsc_type = self.demitionLayer2(sclist[:,-1,:,512:535])
        nextsc_loc = self.demitionLayer3(sclist[:,-1,:,537:539])
        nextsc = torch.concat((nextsc_text,nextsc_type),2)+nextsc_loc
        nextsc = self.pos(nextsc)
        
        next_screen = self.transformer_sc(nextsc.transpose(1,0))#(objs , batch , dim)
        
        upper_input_next = torch.zeros(next_screen.shape[0] , next_screen.shape[1] , upper_tmp_next.shape[2]).to(device)#(objs , batch , dim)
        for iobj in range(next_screen.shape[0]):
            for ibatch in range(next_screen.shape[1]) :
                upper_input_next[iobj][ibatch] = upper_tmp_next[int(next_target[iobj][ibatch].item())][ibatch]
                
        upper_and_next = torch.cat([upper_input_next,next_screen],dim = 2)
        out_next = self.transformer_seq(seq_input.transpose(1,0))#(batch , newdim)
        
        next_context_w= self.demitionLayer_day(contextlist[:,-1,:7])
        next_context_r= self.demitionLayer_hour(contextlist[:,-1,7:31])
        next_context_i= self.demitionLayer_interval(contextlist[:,-1,31:32])
        next_context_app= self.demitionLayer_app(contextlist[:,-1,32:33])
        next_context=torch.concat((next_context_w,next_context_r,next_context_app),1).unsqueeze(0).repeat(next_screen.shape[0],1,1)
        next_screen = torch.cat([upper_and_next,next_context],axis = 2) + next_context_i
        mlp_input = torch.cat([out_next.unsqueeze(0).repeat(next_screen.shape[0],1,1), next_screen], 2)
        out = self.mlp(mlp_input)
        return out.transpose(1,0)