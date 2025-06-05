import torch
import torch.nn as nn

import torch.nn.functional as F
import torch_geometric.nn as gnn

# Hyperparameters
window_size = 14
batch_size = 32
epochs = 20
lr = 0.001

class CNNRegressor(nn.Module):
    def __init__(self, input_channels, output_size,nom_station,window_size,name="CNN"):
        super(CNNRegressor, self).__init__()
        self.nom = name
        self.nom_station = nom_station
        self.window_size = window_size
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (self.window_size - 4), 50),
            nn.ReLU(),
            nn.Linear(50, output_size)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, features, time]
        x = self.cnn(x)
        x = self.fc(x)
        return x

# Transformer + FC model
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim,window_size, model_dim, num_heads, num_layers,nom_station,dropout=0.1,name="Transformer"):
        super(TransformerRegressor, self).__init__()
        self.nom = name
        self.nom_station = nom_station
        self.model_dim = model_dim
        self.input_fc = nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(model_dim * window_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_fc(x)  # [batch, seq_len, model_dim]
        x = x.permute(1, 0, 2)  # [seq_len, batch, model_dim]
        x = self.transformer(x)  # [seq_len, batch, model_dim]
        x = x.permute(1, 0, 2)  # [batch, seq_len, model_dim]
        return self.output_fc(x)

# Modèle GNN-FCNN
class GNN_FCNN_Regressor(nn.Module):
    def __init__(self, num_nodes, num_features,nom_station,name="GNN", hidden_dim=64, output_size=1):
        super(GNN_FCNN_Regressor, self).__init__()
        self.nom = name
        self.nom_station = nom_station
        self.gcn1 = gnn.GCNConv(num_features, hidden_dim)
        self.gcn2 = gnn.GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * num_nodes, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = x.view(1, -1)  # flatten all node features
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Modele_Serie_Ameliore_Edge_Weight(nn.Module):
    def __init__(self, input_dim, num_nodes,nom_station,name="Model_Serie", model_dim=64, num_heads=4, 
                 num_layers=2, hidden_dim=64, output_size=1, dropout=0.1):
        super().__init__()
        self.nom = name
        self.nom_station = nom_station
        self.model_dim = model_dim
        self.num_nodes = num_nodes

        # Couche d'entrée
        self.input_fc = nn.Linear(input_dim, model_dim)

        # Encodage positionnel appris
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, model_dim))  # max_len=100

        # Transformer pour la modélisation temporelle
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # GCN pour le traitement spatial
        self.gcn1 = gnn.GCNConv(model_dim, hidden_dim)
        self.gcn2 = gnn.GCNConv(hidden_dim, hidden_dim)

        # Fully connected
        self.fc1 = nn.Linear(hidden_dim * num_nodes, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x_seq, edge_index, edge_weight):
        # x_seq: [batch_size, window_size, num_nodes, input_dim]
        batch_size, window_size, num_nodes, input_dim = x_seq.shape

        # Appliquer la FC d'entrée
        x_seq = self.input_fc(x_seq)  # [batch, time, nodes, model_dim]

        # Préparer pour le Transformer : [batch * num_nodes, time, model_dim]
        x_seq = x_seq.permute(0, 2, 1, 3).contiguous()  # [batch, nodes, time, model_dim]
        x_seq = x_seq.reshape(batch_size * num_nodes, window_size, self.model_dim)

        # Ajouter positional encoding tronqué à window_size
        x_seq = x_seq + self.pos_embedding[:, :window_size, :]

        # Passage dans le Transformer
        x_seq = self.transformer(x_seq)  # [batch*num_nodes, time, model_dim]

        # Prendre la dernière étape temporelle
        x_last = x_seq[:, -1, :]  # [batch*num_nodes, model_dim]

        # GCN - adapter edge_index pour le batch
        edge_index_expanded = edge_index.repeat(1, batch_size)
        offsets = torch.arange(batch_size, device=x_seq.device) * num_nodes
        edge_index_expanded = edge_index_expanded + offsets.repeat_interleave(edge_index.size(1)).view(1, -1)

        # Adapter edge_weight pour le batch
        edge_weight_expanded = edge_weight.repeat(batch_size)

        # GCN avec edge_weight
        x_gcn = F.relu(self.gcn1(x_last, edge_index_expanded, edge_weight_expanded))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index_expanded, edge_weight_expanded))
        x_gcn = x_gcn.reshape(batch_size, -1)  # [batch_size, num_nodes * hidden_dim]

        # Fully connected
        x = F.relu(self.fc1(x_gcn))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x