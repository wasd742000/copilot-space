import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch.nn import Module, ModuleList, Conv1d, Sequential, ReLU, Dropout
from torch_geometric.nn import Linear, EdgeConv, GATv2Conv, SAGEConv, BatchNorm
import os
import glob
import yaml

class DilatedResidualLayer(Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1x1 = Conv1d(out_channels, out_channels, kernel_size=1)
        self.relu = ReLU()
        self.dropout = Dropout()

    def forward(self, x):
        out = self.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


# This is for the iterative refinement (we refer to MSTCN++: https://github.com/sj-li/MS-TCN2)
class Refinement(Module):
    def __init__(self, final_dim, num_layers=10, interm_dim=64):
        super(Refinement, self).__init__()
        self.conv_1x1 = Conv1d(final_dim, interm_dim, kernel_size=1)
        self.layers = ModuleList([DilatedResidualLayer(2**i, interm_dim, interm_dim) for i in range(num_layers)])
        self.conv_out = Conv1d(interm_dim, final_dim, kernel_size=1)

    def forward(self, x):
        f = self.conv_1x1(x)
        for layer in self.layers:
            f = layer(f)
        out = self.conv_out(f)
        return out

class SPELL(Module):
    def __init__(self, cfg):
        super(SPELL, self).__init__()
        self.use_spf = cfg['use_spf'] # whether to use the spatial features
        self.use_ref = cfg['use_ref']
        self.num_modality = cfg['num_modality']
        channels = [cfg['channel1'], cfg['channel2']]
        final_dim = cfg['final_dim']
        num_att_heads = cfg['num_att_heads']
        dropout = cfg['dropout']

        if self.use_spf:
            self.layer_spf = Linear(-1, cfg['proj_dim']) # projection layer for spatial features

        self.layer011 = Linear(-1, channels[0])
        if self.num_modality == 2:
            self.layer012 = Linear(-1, channels[0])

        self.batch01 = BatchNorm(channels[0])
        self.relu = ReLU()
        self.dropout = Dropout(dropout)

        self.layer11 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch11 = BatchNorm(channels[0])
        self.layer12 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch12 = BatchNorm(channels[0])
        self.layer13 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch13 = BatchNorm(channels[0])

        if num_att_heads > 0:
            self.layer21 = GATv2Conv(channels[0], channels[1], heads=num_att_heads)
        else:
            self.layer21 = SAGEConv(channels[0], channels[1])
            num_att_heads = 1
        self.batch21 = BatchNorm(channels[1]*num_att_heads)

        self.layer31 = SAGEConv(channels[1]*num_att_heads, final_dim)
        self.layer32 = SAGEConv(channels[1]*num_att_heads, final_dim)
        self.layer33 = SAGEConv(channels[1]*num_att_heads, final_dim)

        if self.use_ref:
            self.layer_ref1 = Refinement(final_dim)
            self.layer_ref2 = Refinement(final_dim)
            self.layer_ref3 = Refinement(final_dim)


    def forward(self, x, edge_index, edge_attr, c=None):
        feature_dim = x.shape[1]

        if self.use_spf:
            x_visual = self.layer011(torch.cat((x[:, :feature_dim//self.num_modality], self.layer_spf(c)), dim=1))
        else:
            x_visual = self.layer011(x[:, :feature_dim//self.num_modality])

        if self.num_modality == 1:
            x = x_visual
        elif self.num_modality == 2:
            x_audio = self.layer012(x[:, feature_dim//self.num_modality:])
            x = x_visual + x_audio

        x = self.batch01(x)
        x = self.relu(x)

        edge_index_f = edge_index[:, edge_attr<=0]
        edge_index_b = edge_index[:, edge_attr>=0]

        # Forward-graph stream
        x1 = self.layer11(x, edge_index_f)
        x1 = self.batch11(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.layer21(x1, edge_index_f)
        x1 = self.batch21(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)

        # Backward-graph stream
        x2 = self.layer12(x, edge_index_b)
        x2 = self.batch12(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.layer21(x2, edge_index_b)
        x2 = self.batch21(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)

        # Undirected-graph stream
        x3 = self.layer13(x, edge_index)
        x3 = self.batch13(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)
        x3 = self.layer21(x3, edge_index)
        x3 = self.batch21(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)

        x1 = self.layer31(x1, edge_index_f)
        x2 = self.layer32(x2, edge_index_b)
        x3 = self.layer33(x3, edge_index)

        out = x1+x2+x3

        if self.use_ref:
            xr0 = torch.permute(out, (1, 0)).unsqueeze(0)
            xr1 = self.layer_ref1(torch.softmax(xr0, dim=1))
            xr2 = self.layer_ref2(torch.softmax(xr1, dim=1))
            xr3 = self.layer_ref3(torch.softmax(xr2, dim=1))
            out = torch.stack((xr0, xr1, xr2, xr2), dim=0).squeeze(1).transpose(2, 1).contiguous()

        return out


class GraphDataset(Dataset):
    """
    General class for graph dataset
    """

    def __init__(self, path_graphs):
        super(GraphDataset, self).__init__()
        self.all_graphs = sorted(glob.glob(os.path.join(path_graphs, '*.pt')))

    def len(self):
        return len(self.all_graphs)

    def get(self, idx):
        data = torch.load(self.all_graphs[idx])
        return data

def get_cfg(args):
    """
    Initialize the configuration given the optional command-line arguments
    """
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
        delattr(args, 'cfg')

    for k, v in vars(args).items():
        if v is None and k in cfg:
            continue

        cfg[k] = v

    return cfg

def build_model(cfg, device):
    """
    Build the model corresponding to "model_name"
    """
    model_name = cfg['model_name']
    model = globals()[model_name](cfg)

    return model.to(device)

def get_formatting_data_dict(cfg):
    """
    Get a dictionary that is used to format the results following the formatting rules of the evaluation tool
    """
    data_dict = {}
    return data_dict


def get_formatted_preds(logits, g):
    """
    Get a list of formatted predictions from the model output, which is used to compute the evaluation score
    """
    preds = []
    
    tmp = logits
    tmp = torch.sigmoid(tmp.squeeze().cpu()).numpy().tolist()
    (g,) = g
    preds.append([f"video_{g}", tmp])

    return preds

def predict_single(cfg, data_path, model_path):
    """
    Run prediction on single data
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Build and load model
    model = build_model(cfg, device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Prepare data loader and formatting dictionary
    data_loader = DataLoader(GraphDataset(data_path))
    data_dict = get_formatting_data_dict(cfg)
    
    # Run prediction
    predictions = []
    with torch.no_grad():
        for data in data_loader:
            g = data.g.tolist()
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            c = None
            if cfg['use_spf']:
                c = data.c.to(device)

            logits = model(x, edge_index, edge_attr, c)
            preds = get_formatted_preds(logits, g)
            predictions.extend(preds)
            
    return predictions

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data', type=str, default='./data', help='Root directory to the data')
    parser.add_argument('--root_result', type=str, default='./results', help='Root directory to output')
    parser.add_argument('--data_path', type=str, required=True, help='Path to input data')
    parser.add_argument('--cfg', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint file')

    args = parser.parse_args()
    cfg = get_cfg(args)
    predictions = predict_single(cfg, args.data_path, args.model_path)
    print("Formatted Predictions:", predictions)