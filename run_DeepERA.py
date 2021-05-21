import pickle
import sys
import os
from timeit import default_timer as timer
import copy
import numpy as np
import pandas as pd
import scipy.sparse as scs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, average_precision_score

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
"""CPU or GPU."""
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# find the index of element in lst to targ list
def transIndex(lst, targ):
    idx = torch.zeros_like(lst, dtype=torch.long, device=device)
    for i,ele in enumerate(targ):
        idx[lst==ele] = i
    return idx

# mini-batching
def travInvolve(dest_c,dest_p,layer):
    Ns_c = dest_c.clone() # list of destination compounds in the current layer
    Ns_p = dest_p.clone() # list of destination proteins in the currect layer

    for ix in reversed(range(layer)): # start with top layer and reversely extract nodes and edges
    # here the reversed function will not make a difference, but I use this way to indicate that we are tracing from the top level
        # for drugs
        i_c = torch.zeros(net_c[1,:].size(0), dtype=torch.bool, device=device) # use logic data, all false at default
        # extract all nodes in the previous layer pointing to the nodes in the current layer
        for ele in Ns_c:
            i_c = i_c | (net_c[1,:]==ele) # net_c[1,:] stores the destination nodes        
        Es_c = net_c[:,i_c] # extract edges
        Ns_c = Es_c.unique() # obtain the node list of the previous layer
        # for proteins
        i_p = torch.zeros(net_p[1,:].size(0),dtype=torch.bool, device = device)
        for ele in Ns_p:
            i_p = i_p | (net_p[1,:]==ele)
        Es_p = net_p[:,i_p]
        Ns_p = Es_p.unique()
    return Ns_c, Ns_p

class DeepERA(nn.Module):
    def __init__(self):
        super(DeepERA, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_fea_d = nn.ModuleList([nn.Linear(dim_d, 2*dim), nn.Linear(2*dim, dim), nn.Linear(dim, dim)])
        self.W_fea_p = nn.ModuleList([nn.Linear(dim_p, 2*dim), nn.Linear(2*dim, dim), nn.Linear(dim, dim)])
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_gcn_d = nn.ModuleList([nn.Linear(dim_c, dim)
                                    for _ in range(layer_gcn)])
        self.W_gcn_p = nn.ModuleList([nn.Linear(dim_t, dim)
                                    for _ in range(layer_gcn)])
        self.W_out = nn.ModuleList([nn.Linear(6*dim, 4*dim), nn.Linear(4*dim, 4*dim), nn.Linear(4*dim, 4*dim)])
        self.W_interaction = nn.Linear(4*dim, 2)

    def lnn_d(self, xs, layer):
        """Annotation embedding for drugs"""
        for i in range(layer):
            xs = torch.relu(self.W_fea_d[i](xs))
        return xs

    def lnn_p(self, xs, layer):
        """Annotation embedding for proteins"""
        for i in range(layer):
            xs = torch.relu(self.W_fea_p[i](xs))
        return xs

    def gnn(self, xs, A, layer):
        """Intrinsic embedding based on GNN for drugs"""
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def cnn(self, xs, layer):
        """Intrinsic embedding based on CNN for proteins"""
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        ys = torch.unsqueeze(torch.mean(xs, 0), 0)
        return ys

    # Relational embedding
    def gcn_d(self, idx, layer):
        """GNN for proteins
        Xs_c: whole set of input feature vectors (deduplicated), global
        idx: indices of data points
        A_c: adjacency matrix (contains self-loop), global
        layer: number of layers in GNN
        """
        Xs = Xs_c.clone() # here use clone to create a copy as we will change the variable 'Xs' later
        for i in range(layer):
            hs = torch.matmul(A_c, Xs) # message propagation and aggregation
            Xs = torch.relu(self.W_gcn_d[i](hs)) # pass through neural network: Wx+b
        return Xs[idx,:] # return the embedding for nodes only involved in the current iteraction

    def gcn_p(self, idx,layer):
        """GNN for proteins
        Xs_p: whole set of input feature vectors (deduplicated), global
        idx: indices of data points
        A_p: adjacency matrix (contains self-loop), global
        layer: number of layers in GNN
        """
        Xs = Xs_p.clone() # here use clone to create a copy as we will change the variable 'Xs' later
        for i in range(layer):
            hs = torch.matmul(A_p, Xs) # message propagation and aggregation
            Xs = torch.relu(self.W_gcn_p[i](hs)) # pass through neural network: Wx+b
        return Xs[idx,:] # return the embedding for nodes only involved in the current iteraction
            
    def forward(self, idx_c, idx_p):
        words_lst = []
        compound_vector = []
        protein_vector = []
        compound_fingerprints_vector = []
        protein_words_vector = []

        ns_p = idx_p.unique()
        ns_c = idx_c.unique()

        ni_c, ni_p = travInvolve(ns_c,ns_p,layer_gcn) 

        Xs_c_int = torch.zeros([N_c,dim], dtype=torch.float).to(device) # intrinsic embedding of structural info
        Xs_p_int = torch.zeros([N_p,dim], dtype=torch.float).to(device) # 

        for inp in ni_c:
            fingerprints, adjacency = compounds[inp],adjacencies[inp]
            fingerprint_vectors = self.embed_fingerprint(fingerprints)
            Xs_c_int[inp,:] = self.gnn(fingerprint_vectors, adjacency, layer_gnn)
        for inp in ni_p:
            word_vectors = self.embed_word(proteins[inp])
            Xs_p_int[inp,:] = self.cnn(word_vectors, layer_cnn)

        compound_vector_intrinsic = Xs_c_int[idx_c,:]
        protein_vector_intrinsic = Xs_p_int[idx_p,:]

        compound_vector_relational = self.gcn_d(idx_c, layer_gcn) 
        protein_vector_relational = self.gcn_p(idx_p, layer_gcn) 

        compound_features =  self.lnn_d(drug_feat[idx_c,:],3)
        protein_features =  self.lnn_p(protein_feat[idx_p,:],3)
        cat_vector = torch.cat((compound_vector_intrinsic,compound_vector_relational,compound_features,protein_vector_intrinsic,protein_vector_relational,protein_features), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return torch.sigmoid(interaction)

    def __call__(self, input_data):

        idx_c = input_data[:,0]
        idx_p = input_data[:,1]
        
        correct_interaction = input_data[:,2].view(-1,1) # make it 2D, which will be used to spread correct labels
        predicted_interaction = self.forward(idx_c, idx_p)

        correct_labels = list(correct_interaction.to('cpu').data.numpy())
        ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
        predicted_labels = list(map(lambda x: np.argmax(x), ys))
        predicted_scores = list(map(lambda x: x[1], ys))
            
        y_onehot = torch.FloatTensor(len(idx_c), 2).to(device)
        y_onehot.zero_()
        y_onehot.scatter_(1, correct_interaction, 1)
        
        loss = F.binary_cross_entropy(predicted_interaction, y_onehot)
        return loss,correct_labels, predicted_scores

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy')]

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":

    """Hyperparameters."""
    (batch_size, radius, ngram, dim, layer_gnn, layer_gcn, window, layer_cnn, layer_output,
     lr, lr_decay, decay_interval, weight_decay, epochs) = sys.argv[1:]
    (batch_size, dim, layer_gnn, layer_gcn, window, layer_cnn, layer_output, decay_interval,
     epochs) = map(int, [batch_size, dim, layer_gnn, layer_gcn, window, layer_cnn, layer_output,
                            decay_interval, epochs])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    """Load processed data."""
    dir_input = ('Processed_data/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)

    idx_c = np.load(dir_input + 'compounds_node_index.npy').reshape([-1,1])
    idx_p = np.load(dir_input + 'proteins_node_index.npy').reshape([-1,1])    
    tort = np.load(dir_input + 'trainortest.npy')
    inter = np.load(dir_input + 'interactions.npy')
    all_dat = np.hstack([tort,idx_c,idx_p,inter]).astype(int) # where t_or_t indicate if the data sample belongs to training (1), validation (2), or testing (3) set
    # separate training, validation, and testing set
    train_set = all_dat[all_dat[:,0]==1,1:] # remove the first column
    validate_set = all_dat[all_dat[:,0]==2,1:]
    test_set = all_dat[all_dat[:,0]==3,1:]
    train_set = torch.LongTensor(train_set).to(device)
    validate_set = torch.LongTensor(validate_set).to(device)
    test_set = torch.LongTensor(test_set).to(device)

    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    # load networks and remove self-loop
    net_c = np.load(dir_input + 'compounds_network.npy').T
    net_p = np.load(dir_input + 'proteins_network.npy').T
    net_p = net_p[:,net_p[0,:]!=net_p[1,:]]
    net_c = net_c[:,net_c[0,:]!=net_c[1,:]]
    net_c = torch.LongTensor(net_c).to(device)
    net_p = torch.LongTensor(net_p).to(device)

    N_c = int(net_c.max()+1) # Total number of drugs involved
    N_p = int(net_p.max()+1) # Total number of proteins involved

    dim_c = N_c+500
    dim_t = N_p+500

    ls_tr,ls_ts,ls_val,auc_tr,auc_val,auc_ts=[],[],[],[],[],[]
    # create the identify features for all drugs, and make it global
    Xs_c = torch.zeros([N_c,dim_c], dtype=torch.float).to(device) # identity feature matrix for the drugs in batch, e.g. 1000 x 4
    for i in range(N_c):
        Xs_c[i,i] = 1.0

    # create the identify features for all proteins, and make it global
    Xs_p = torch.zeros([N_p,dim_t], dtype=torch.float).to(device) # identity feature matrix for the drugs in batch, e.g. 1000 x 4
    for i in range(N_p):
        Xs_p[i,i] = 1.0


    # load annotations of drugs and proteins
    drug_feat = np.load(dir_input + 'drug_sideeffect_feature.npy')
    protein_feat = np.load(dir_input + 'protein_disease_feature.npy')

    drug_domain = np.load(dir_input + 'drug_domain_feature.npy')
    protein_domain = np.load(dir_input + 'protein_domain_feature.npy')

    drug_feat = np.hstack((drug_feat,drug_domain))
    protein_feat = np.hstack((protein_feat,protein_domain))

    # remove the columns containing all '0' or '1'
    drug_feat = np.delete(drug_feat,np.max(drug_feat,axis=0)==0,1)
    drug_feat = np.delete(drug_feat,np.min(drug_feat,axis=0)==1,1)
    protein_feat = np.delete(protein_feat,np.max(protein_feat,axis=0)==0,1)
    protein_feat = np.delete(protein_feat,np.min(protein_feat,axis=0)==1,1)

    dim_d = drug_feat.shape[1]
    dim_p = protein_feat.shape[1]

    drug_feat = torch.FloatTensor(drug_feat).to(device)
    protein_feat = torch.FloatTensor(protein_feat).to(device)
    drug_domain = torch.FloatTensor(drug_domain).to(device)
    protein_domain = torch.FloatTensor(protein_domain).to(device)


    # construct the adjacency matrix for drugs, and make it global
    A_c = torch.eye(N_c,dtype=torch.float,device=device) # the adjacency matrix for all drugs
    A_c[net_c[0,:],net_c[1,:]] = 1.0 # add edges into the adjacency matrix
    A_c[net_c[1,:],net_c[0,:]] = 1.0 # undirected edges

    # construct the adjacency matrix for drugs, and make it global
    A_p = torch.eye(N_p,dtype=torch.float,device=device) # the adjacency matrix for all proteins
    A_p[net_p[0,:],net_p[1,:]] = 1.0 # add edges into the adjacency matrix
    A_p[net_p[1,:],net_p[0,:]] = 1.0 # undirected edges



    m_e = False
    fnam = 'models/Pretrained.pickle'
#     try:
#         model = torch.load(fnam)
#         model.eval()
#         m_e = True
#     except:
#         print(f'No pre-trained model {fnam} is found!')

    if not m_e:
        model = DeepERA().to(device)

    
    optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)

    print('Epoch\tTime\ttrain_loss\ttrain_AUC\tvalidate_AUC\ttest_AUC')
    for epoch in range(epochs):
        if epoch % decay_interval == 0:
            optimizer.param_groups[0]['lr'] *= lr_decay

        train_set=train_set[torch.randperm(train_set.shape[0]),:] # shuffle the data points in each epoch
        N = train_set.shape[0] # total number of DPI pairs in the training set
        AUC_lst, loss_lst = [],[]
        all_correct, all_predict = [],[] # list of correct labels/ predicted scores for the whole training set
        start = timer()
        train_loss = 0
        for i in np.arange(0,N,batch_size):
            if N - i < batch_size:
                raw_dat = train_set[i:]
            else:
                raw_dat = train_set[i:i+batch_size]

            loss, correct_label, predict_score = model(raw_dat)
            train_loss += loss.item()
            all_correct += correct_label # append the labels in the batch to the whole list
            all_predict += predict_score
            if not m_e: # if the model is not pre-trained, train it
                optimizer.zero_grad() # initialize the gradient
                loss.backward()
                optimizer.step()
            del loss, correct_label, predict_score
        torch.cuda.empty_cache()
        elapsed_time = timer() - start
        train_AUC = round(roc_auc_score(all_correct, all_predict),4) # calculate AUC for the whole training set
        del all_correct, all_predict

        N = validate_set.shape[0]
        correct_label, predict_score = [],[]
        for i in np.arange(0,N,batch_size*10):
            if N - i < batch_size*10:
                raw_dat = validate_set[i:]
            else:
                raw_dat = validate_set[i:i+batch_size*10]
            lss, correct_l, predict_s = model(raw_dat)
            correct_label += correct_l
            predict_score += predict_s
            del lss, correct_l, predict_s
        validate_AUC = round(roc_auc_score(correct_label, predict_score),4) # calculate AUC for the whole validation set

        del correct_label, predict_score
        torch.cuda.empty_cache()

        N = test_set.shape[0]
        correct_label, predict_score = [],[]
        for i in np.arange(0,N,batch_size*10):
            if N - i < batch_size*10:
                raw_dat = test_set[i:]
            else:
                raw_dat = test_set[i:i+batch_size*10]
            lss, correct_l, predict_s = model(raw_dat)
            correct_label += correct_l
            predict_score += predict_s
            del lss, correct_l, predict_s
        test_AUC = round(roc_auc_score(correct_label, predict_score),4) # calculate AUC for the whole test set
        del correct_label, predict_score
        torch.cuda.empty_cache()

        
        auc_tr.append(train_AUC)
        auc_ts.append(test_AUC)
        auc_val.append(validate_AUC)
        print(f'{epoch}\t{elapsed_time:.4f}\t{train_loss:.4f}\t{train_AUC:.4f}\t{validate_AUC:.4f}\t{test_AUC:.4f}')
        del train_AUC, test_AUC, validate_AUC
        if m_e:
            break

#    torch.save(model,fnam)
