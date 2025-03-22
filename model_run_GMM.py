from sklearn.metrics import roc_auc_score 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from parse import parse_method
from torch.distributions import MultivariateNormal  
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score  



class GMMLoss(nn.Module):
    def __init__(self, num_center=1, feat_dim=32, Mahalanobis='diagonal_nequal', Measurement='closest', use_gpu=True):
        super(GMMLoss, self).__init__()
        self.num_center = num_center
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.measurement = Measurement

        if Mahalanobis == 'base': 
            self.raw_weights = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        elif Mahalanobis == 'diagonal_equal':  
            self.raw_weights = nn.ParameterList([nn.Parameter(torch.tensor(1.0, requires_grad=True)) for _ in range(num_center)])  
        elif Mahalanobis == 'diagonal_nequal':   
            self.raw_weights = nn.ParameterList([nn.Parameter(torch.ones(feat_dim, requires_grad=True) * (1.0/feat_dim)) for _ in range(num_center)])  
        elif Mahalanobis == 'matrix':  
            self.raw_weights = nn.ParameterList([nn.Parameter(torch.tril(torch.ones(feat_dim, feat_dim))) for _ in range(num_center)])  

        if self.use_gpu:
            self.centers = torch.randn(self.num_center, feat_dim, device='cuda')
        else:
            self.centers = torch.randn(self.num_center, feat_dim)
        
        if Measurement == 'closest':
            self.measurement = 'closest'
        else:
            self.measurement == 'mixed'
            self.mixing_coeffs = torch.ones(num_center) / num_center
        self.update_responsibilities = False

    def compute_covariances(self):  
        covariances = []  
        if isinstance(self.raw_weights, nn.Parameter):  
            # base case: scalar weight  
            for _ in range(self.num_center):  
                covariances.append(torch.eye(self.feat_dim, device=self.raw_weights.device) * self.raw_weights)  
        elif isinstance(self.raw_weights, nn.ParameterList):  
            for i, weight in enumerate(self.raw_weights):  
                if weight.dim() == 0:  
                    # diagonal_equal case: scalar weight for each center  
                    covariances.append(torch.eye(self.feat_dim, device=weight.device) * weight)  
                elif weight.dim() == 1:  
                    # diagonal_nequal case: diagonal weights for each center
                    normalized_weights = F.softmax(F.softplus(weight), dim=0)   
                    covariances.append(torch.diag(normalized_weights))  
                elif weight.dim() == 2:  
                    # matrix case: lower triangular matrix for each center  
                    tril_matrix = torch.tril(weight)  
                    covariances.append(tril_matrix @ tril_matrix.t())  
        return covariances
    
    def compute_responsibilities(self, x):  
        covariances = self.compute_covariances()
        # Calculate responsibilities (weights) for each center  
        log_responsibilities = torch.zeros(x.size(0), self.num_center, device=x.device)  
        for i in range(self.num_center):  
            mvn = MultivariateNormal(self.centers[i], covariances[i])  
            log_responsibilities[:, i] = torch.log(self.mixing_coeffs[i]) + mvn.log_prob(x) 
        
        # Normalize to get responsibilities  
        log_responsibilities -= torch.logsumexp(log_responsibilities, dim=1, keepdim=True)  
        self.responsibilities = log_responsibilities.exp()  

    def update_mixing_coefficients(self):  
        N_k = self.responsibilities.sum(dim=0)  # Sum of responsibilities for each component  
        N = self.responsibilities.size(0)  # Total number of data points  
        self.mixing_coeffs = N_k / N  # Update mixing coefficients 

    def set_update_responsibilities(self):
        self.update_responsibilities = True

    def forward(self, x):

        if self.update_responsibilities:
            with torch.no_grad(): 
                self.compute_responsibilities(x)
                self.update_mixing_coefficients()
                self.update_responsibilities = False
        

        batch_size = x.size(0)
        distmat = torch.zeros(batch_size, self.num_center)  
        if isinstance(self.raw_weights, nn.ParameterList) and self.raw_weights[0].dim() == 2:  
            for i in range(self.num_center):  
                with torch.no_grad():  
                    raw_weights = torch.tril(self.raw_weights[i])  
                mahalanobis_matrix = raw_weights @ raw_weights.t()  

                flat_weights = mahalanobis_matrix.view(-1)  
                normalized_weights = F.softmax(F.softplus(flat_weights), dim=0)  
 
                mahalanobis_matrix = normalized_weights.view(mahalanobis_matrix.size())  

                diff = x - self.centers[i]  
                distmat[:, i] = torch.sum(diff @ mahalanobis_matrix * diff, dim=1)  

        elif isinstance(self.raw_weights, nn.ParameterList):
            for i in range(self.num_center):  
                diag_weights = self.raw_weights[i]  
                normalized_weights = F.softmax(F.softplus(diag_weights), dim=0)  

                diff = (x - self.centers[i]) * normalized_weights  
                distmat[:, i] = torch.sum(diff * diff, dim=1)  
 
        else:
            normalized_weights = F.softmax(F.softplus(self.raw_weights), dim=0)
            x = x * normalized_weights  
            centers = nn.Parameter(self.centers * normalized_weights)
            distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_center) + \
                    torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(self.num_center, batch_size).t()
            distmat.addmm_(1, -2, x, centers.t())
        # distmat = torch.exp(-0.5 * distmat)
             

        # classes = torch.arange(self.num_center).long()
        # if self.use_gpu: classes = classes.cuda()
        # labels = labels.unsqueeze(1).expand(batch_size, self.num_center)
        # mask = labels.eq(classes.expand(batch_size, self.num_center))

        # dist = distmat * mask.float()
        if self.measurement == 'closest':
            distmat, _ = torch.min(distmat, dim=1)
        else: 
            mixing_coeffs = self.mixing_coeffs.to(distmat.device)  
            # distmat = torch.exp(-0.5 * distmat)
            distmat = torch.sum(distmat * mixing_coeffs.unsqueeze(0), dim=1) 

        
        dist = distmat
        return dist

def Train(model, graph, features, train_idx, svdd_loss, optimizer):
    model.train()
    optimizer.zero_grad()
    outputs = model(graph, features)

    dist= svdd_loss(outputs[train_idx])

    loss = torch.mean(dist)

    loss.backward() 
    optimizer.step() 
    return loss.item() 

def Evaluate(model, graph, features, labels, svdd_loss, mask):
    model.eval()  
    with torch.no_grad():  
        outputs = model(graph, features) 

        dist= svdd_loss(outputs[mask])

        labels = labels[mask].cpu().numpy()
        scores = dist.cpu().numpy()
        auc = roc_auc_score(labels, scores)
        # preds = np.argmax(logits, axis=1)   
        threshold = np.percentile(scores, 85)  
 
        predictions = (scores >= threshold).astype(int)  

        f1_positive = f1_score(labels, predictions)  
        precision_positive = precision_score(labels, predictions)  
        recall_positive = recall_score(labels, predictions)  


        predictions_negative = (scores < threshold).astype(int) 
        labels_negative = 1 - labels  

        f1_negative = f1_score(labels_negative, predictions_negative)  
        precision_negative = precision_score(labels_negative, predictions_negative)  
        recall_negative = recall_score(labels_negative, predictions_negative)  

        macro_f1 = (f1_positive + f1_negative) / 2
        acc = accuracy_score(labels, predictions) 
        g_mean = np.sqrt(recall_positive * recall_negative)  
 

    return [auc, acc, macro_f1, precision_positive, precision_negative, recall_positive, recall_negative, g_mean]


def GMM_mod_run(args, device, dataset, out_dim=32):
    graph = dataset[0]

    # graph = graph.subgraph([i for i in range(10000)])

    
    labels = graph.ndata["label"].to(device)
    feat = graph.ndata["feature"].to(device)

    train_mask = graph.ndata["train_mask"].to(device)
    val_mask = graph.ndata["val_mask"].to(device)
    test_mask = graph.ndata["test_mask"].to(device)

    train_idx = torch.nonzero((train_mask & (labels == 0)).to(device), as_tuple=False).squeeze(1).to(device)
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze(1).to(device)
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze(1).to(device)

    graph = graph.to(device)

    n = len(graph.ndata["label"])
    c = out_dim
    d = graph.ndata["feature"].shape[1]

    train_pos = []

    model = parse_method(args, dataset, n, c, d, train_pos, device)

    print('MODEL:', model)
    print('DATASET:', args.dataset)

    model.reset_parameters()


    # Define loss function and optimizer  
    svdd_loss = GMMLoss(num_center=args.gmm_K, feat_dim=out_dim, 
                             Mahalanobis=args.Mahalanobis, Measurement=args.Measurement).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(svdd_loss.parameters()), lr=args.lr) 
    patience = 150  
    best_val_auc = 0.0  
    epochs_no_improve = 0 
    best_model_path = f"best_model_{device}.pth"  
    best_weights_path = f"best_weights_{device}.pth"  

    if args.Measurement == 'mixed':
        svdd_loss.set_update_responsibilities()

    # Training loop  
    for epoch in range(args.epochs):  
        loss = Train(model, graph, feat, train_idx, svdd_loss, optimizer) 
        val_auc = Evaluate(model, graph, feat, labels, svdd_loss, val_idx)
        # val_auc = 0

        if args.early_stop:
            if val_auc[0] > best_val_auc:  
                best_val_auc = val_auc[0]  
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)   
                torch.save(svdd_loss.state_dict(), best_weights_path) 
            else:  
                epochs_no_improve += 1  
            if epochs_no_improve == patience:  
                print(f"Early stopping at epoch {epoch+1}")  
                break 

        if (epoch + 1) % 25 == 0:
            if args.Measurement == 'mixed':
                svdd_loss.set_update_responsibilities()  
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}, Validation AUC: {val_auc[0]:.4f}") 

    # Test after training
    model.load_state_dict(torch.load(best_model_path)) 
    svdd_loss.load_state_dict(torch.load(best_weights_path))      
    test_auc = Evaluate(model, graph, feat, labels, svdd_loss, test_idx)  
    print(f"Test AUC: {test_auc[0]:.4f}, Test ACC: {test_auc[1]:.4f}, Test F1: {test_auc[2]:.4f}")  
    print(f"Test Precision_1: {test_auc[3]:.4f}, Test Precision_2: {test_auc[4]:.4f}")  
    print(f"Test Recall_1: {test_auc[5]:.4f}, Test Recall_2: {test_auc[6]:.4f}")  
    print(f"Test G-mean: {test_auc[7]:.4f}")  