##### DEFINITION DU FICHIER UTILS #########

#### Import ####
#Import
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from torch_geometric.utils import add_self_loops
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch_geometric.data import Data

from matplotlib.ticker import MaxNLocator
import torch.nn as nn

from geopy.distance import geodesic
from torch.utils.data import Subset

from datetime import datetime



# Hyperparameters
#window_size = 14
#batch_size = 32
#epochs = 20
#lr = 0.001


def traitement_donnes(nom_station,year_start,year_end):
    """
    À partir du fichier de données, crée les tenseurs qui seront utilisés par les différents modèles.

    Args:
        nom_station (str) : Nom de la stion que l'on souhaité prévoir.

    Returns:
        Tuple[Tensor, Tensor] : Tenseurs d'entrée (x) et de sortie (y) prêts pour l'entraînement.
    """

    
    file_path = 'Data_Q_H_X_S_1977-2020 - ST (1).xlsx'
    df_water = pd.read_excel(file_path, sheet_name='Data Q,H,S', header=1)  
    df_rain = pd.read_excel(file_path, sheet_name='Precipitation data', header=1)  

    if 'Unnamed: 0' in df_rain.columns:
        df_rain = df_rain.rename(columns={'Unnamed: 0': 'Time'})

    # Nettoyage
    df_rain = df_rain.loc[:, ~df_rain.columns.str.startswith('Unnamed')]
    df_water.columns = ['Time', 'Thanh My', 'Nong Son', 'Ai Nghia', 'Cam Le', 'Cau Lau', 'Giao Thuy', 'Hoi An', 'Hoi Khach']
    df_rain.columns = ['Time', '(2)', '(3)', 'Ai Nghia', 'Cam Le', 'Cau Lau', 'Giao Thuy', 'Hien', 'Hiep Duc', 'Hoi An', 'Hoi Khach', 'Kham Duc', 'Nong Son', 'Que Son', 'Thanh My', 'Tien Phuoc', 'Da Nang', 'Tam Ky', 'Tra My']

    # Colonnes communes
    common_columns = list(set(df_water.columns) & set(df_rain.columns))
    common_columns_with_time = ['Time'] + [col for col in common_columns if col != 'Time']
    df_water_common = df_water[common_columns_with_time].copy()
    df_rain_common = df_rain[common_columns_with_time].copy()


    df = pd.merge(df_water_common, df_rain_common, on='Time', suffixes=('_water', '_rain'))
    df = df.sort_values('Time').reset_index(drop=True)
    df = df.ffill().bfill()
    df = df[(df['Time'].dt.year >= year_start) & (df['Time'].dt.year <= year_end)]


    stations = ['Thanh My', 'Nong Son', 'Ai Nghia', 'Cau Lau', 'Giao Thuy', 'Hoi An', 'Hoi Khach']

    # Organisation des colonnes
    ordered_columns = []
    for station in stations:
        if f"{station}_rain" in df.columns:
            ordered_columns.append(f"{station}_rain")
        if f"{station}_water" in df.columns:
            ordered_columns.append(f"{station}_water")

    # Données d'entrée et target
    
    target_station = nom_station
    target_station_water = f"{target_station}_water"
    
    # 1. Exclure explicitement la target des features
    features_columns = [col for col in ordered_columns if col != target_station_water]
    features = df[features_columns]
    features_non_rain = features.drop(f"{target_station}_rain",axis=1)

    target = df[[target_station_water]]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(features)
    X_scaled_non_rain = scaler_X.fit_transform(features_non_rain)
    
    y_scaled = scaler_y.fit_transform(target)

    nombre_stations = 6
    num_features = X_scaled.shape[1]
    num_samples = X_scaled.shape[0] 

    X_reshaped = X_scaled_non_rain.reshape(num_samples, nombre_stations, 2)  
    X_reshaped = torch.tensor(X_reshaped, dtype=torch.float32)

    empty_node = torch.zeros((num_samples, 1, 2))  
    X_with_station = torch.cat([X_reshaped, empty_node], dim=1) 

    return X_scaled,y_scaled,X_with_station,scaler_X,scaler_y,df




# Dataset class
class WaterLevelDataset(Dataset):
    """
    À partir des tenseurs fournis, crée un Dataset utilisable par le modèle.

    Args:
        x (Tensor) : Données d'entrée.
        y (Tensor) : Cibles associées.

    Returns:
        Dataset : Un objet Dataset compatible avec un DataLoader PyTorch.
    """

    def __init__(self, X, y, window_size):
        self.X = []
        self.y = []
        for i in range(window_size, len(X)):
            self.X.append(X[i - window_size:i])
            
            self.y.append(y[i])
            
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def loader(X_scaled,y_scaled,batch_size,window_size):
    """
    Retourne les DataLoader d'entraînement et de validation utilisés pour entraîner et
    valider le modèle.

    Returns:
        tuple: Un tuple contenant deux DataLoader :
            - train_loader (DataLoader) : pour l'entraînement du modèle.
            - val_loader (DataLoader) : pour la validation du modèle.
    """

    # Dataset and DataLoader
    dataset = WaterLevelDataset(X_scaled, y_scaled, window_size)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, remaining = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    val_indices = list(range(train_size, len(dataset)))  # indices après ceux du train
    val_set = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    return train_loader,val_loader


def training_model(model,save, loss_fn,epochs, optimizer, train_loader,model_path=None,edge_index=None,edge_weights=None):
    """
    Entraîne le modèle à partir du `train_loader` en fonction de son nom, afin de respecter
    les spécificités de son architecture.

    Args:
        model (torch.nn.Module): Le modèle à entraîner.
        loss_fn (torch.nn.Module): La fonction de perte.
        epochs (int): Le nombre d'époques d'entraînement.
        optimizer (torch.optim.Optimizer): L'optimiseur utilisé pour l'entraînement.
        train_loader (DataLoader): DataLoader contenant les données d'entraînement.
        model_path (str): Chemin où sauvegarder le modèle entraîné.
        edge_index (torch.Tensor, optional): Indice des arêtes du graphe (si applicable).
        edge_weights (torch.Tensor, optional): Poids des arêtes du graphe (si applicable).

    Returns:
        torch.nn.Module: Le modèle entraîné, également sauvegardé dans le fichier spécifié.
    """

    if model_path !=None:
        model.load_state_dict(torch.load(model_path))
        print(f"Modèle chargé depuis {model_path}")
        return model
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # Pour CNN et Transformer
        if model.nom in ['CNN', 'Transformer']:
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = loss_fn(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

        # Pour GNN / Serie
        elif model.nom == 'GNN':
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss_batch = 0
                for t in range(x_batch.size(0)):  # boucle sur le batch
                    x_input = x_batch[t][-1]  # shape [nb_stations, nb_features]
                    data = Data(x=x_input, edge_index=edge_index)
                    output = model(data.x, data.edge_index)
                    loss = loss_fn(output, y_batch[t])
                    loss.backward()
                    loss_batch += loss.item()
                optimizer.step()
                total_loss += loss_batch
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

        elif model.nom == 'Model_Serie':
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(x_batch, edge_index,edge_weights)
                loss = loss_fn(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    
    if save:

        save_path = rf"C:\Users\roumi\OneDrive\Bureau\Projet_Stage\Models\{model.nom}\Model_{model.nom}_{model.nom_station}_epoch_{epochs}_{timestamp}.pt"
        torch.save(model.state_dict(), save_path)
        print(rf'Le model est enregistré sous le nom: Model_{model.nom}_{model.nom_station}_e={epochs}_{timestamp}.pt'
          )
    return model


# Générer l'edge_index pour PyTorch Geometric
def get_edge_index(adj_matrix):
    """
    Retourne le edge_index à partir de la matrice d'adjacence.
    Décrit la manière dont les stations sont connectées entre elles dans le graphe.

    Args:
    adj_matrix (numpy.ndarray ou torch.Tensor): Matrice d'adjacence représentant les connexions entre stations.

    Returns:
    torch.Tensor: Tensor de forme [2, num_edges] représentant les indices des arêtes.
    """

    edge_list = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] != 0:
                edge_list.append([i, j])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index, _ = add_self_loops(edge_index, num_nodes=adj_matrix.shape[0])
    return edge_index

def create_edge_weights(distance_matrix, edge_index, device='cpu'):
    """
    Calcule les poids des arêtes (edge weights) pour un modèle basé sur un graphe.
    Les stations géographiquement proches reçoivent des poids plus élevés.

    Returns:
    torch.Tensor: Vecteur des poids associés aux arêtes du graphe.
    """

    weights = []
    for i, j in edge_index.t().tolist():
        dist = distance_matrix[i][j]
        weights.append(1.0 / (dist + 1e-6))  # Inverse distance pour éviter div/0

    weights = torch.tensor(weights, dtype=torch.float32)

    # Normalisation entre 0 et 1
    weights = weights / weights.max()

    return weights.to(device)

def get_distance_matrix(adj_matrix):
    """
    Construit une matrice de distance entre stations, utilisée pour calculer les poids des arêtes du graphe.
    """

    # Lire le fichier Excel
    stations_df = pd.read_excel("Co-ordinate Station.xlsx", engine='openpyxl')

    stations = ['Thành Mỹ','Hội Khánh' ,'Ái Nghĩa','Nông Sơn','Giao Thủy', 'Câu Lâu',  'Hội An']
    station_names = stations

    # Filtrer les lignes dont le nom de station est dans la liste
    filtered_df = stations_df[stations_df.iloc[:, 0].isin(stations)]

    # Extraire les coordonnées
    longitudes = filtered_df.iloc[:, 1].astype(float).values
    latitudes = filtered_df.iloc[:, 2].astype(float).values
    print("longitudes = ",longitudes)
    print("latitudes = ",latitudes)

    # 1. Préparation des données géospatiales
    coords = list(zip(latitudes, longitudes))  # Création des paires (lat, lon)

    # 2. Création du dictionnaire de positions géoréférencées
    geo_pos = {i: (lon, lat) for i, (lat, lon) in enumerate(zip(latitudes, longitudes))}  # Inversion pour (x=lon, y=lat)

    # 3. Calcul des distances réelles entre stations (optionnel)
    distance_matrix = np.zeros((len(stations), len(stations)))
    for i in range(len(stations)):
        for j in range(len(stations)):
            if i != j and adj_matrix[i,j] == 1:
                distance_matrix[i,j] = geodesic(coords[i], coords[j]).km
    return distance_matrix

def evaluate_model(model,save,loss_fn,val_loader,scaler_y,df,year_end,month=None,jours_afficher=None,edge_index=None,edge_weights=None):
    """
    Évalue un modèle selon son nom et trace la courbe de comparaison entre les valeurs prédites et réelles.

    Cette fonction permet d'afficher une courbe pour visualiser les performances du modèle.
    Args:
        model: Le modèle entraîné à évaluer.
        loss_fn: La fonction de perte utilisée.
        val_loader: Le DataLoader de validation.
        scaler_y: Le scaler utilisé pour les valeurs cibles (si applicable).
        df: Le DataFrame contenant les données d'origine.
        edge_index: La structure du graphe (indices des arêtes).
        edge_weights: Les poids des arêtes (si applicables).
    """
    # Collect predictions and true values
    predictions = []
    actuals = []
    ## Evaluate the model
    model.eval()

    with torch.no_grad():
        total_loss = 0
        if model.nom in ['CNN', 'Transformer']:
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                loss = loss_fn(output, y_batch)
                total_loss += loss.item()
                predictions.extend(output.squeeze().tolist())
                actuals.extend(y_batch.squeeze().tolist())

        elif model.nom == 'GNN':
            for x_batch, y_batch in val_loader:
                for t in range(x_batch.size(0)):
                    x_input = x_batch[t][-1]  # dernière fenêtre temporelle
                    data = Data(x=x_input, edge_index=edge_index)
                    output = model(data.x, data.edge_index)
                    loss = loss_fn(output, y_batch[t])
                    total_loss += loss.item()
                    predictions.append(output.item())
                    actuals.append(y_batch[t].item())

        elif model.nom == 'Model_Serie':
            for x_batch, y_batch in val_loader:
                output = model(x_batch, edge_index,edge_weights)
                loss = loss_fn(output, y_batch)
                total_loss += loss.item()
                predictions.append(output.numpy())
                actuals.append(y_batch.numpy())
        else:
             raise ValueError(f"Nom de modèle inconnu : {model.nom}")
        
    print(f"Test Loss: {total_loss / len(val_loader):.4f}")
    print("Nombre des predictions = ",len(predictions),"Nombre de actuels",len(actuals))

    if model.nom == "Model_Serie":
        predictions = np.vstack(predictions)
        actuals = np.vstack(actuals)
    
    # Convert predictions and actuals back to numpy arrays
    y_true = np.array(actuals)
    y_pred = np.array(predictions)



    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)

    predictions_inv = scaler_y.inverse_transform(predictions)
    actuals_inv = scaler_y.inverse_transform(actuals)

    y_true = actuals_inv.flatten()
    y_pred = predictions_inv.flatten()

    y_true_return = y_true.copy()
    y_pred_return = y_pred.copy()

    moyenne_annuelle = np.mean(y_true)

    y_true = y_true[(month-1)*30:(month-1)*30+jours_afficher]
    y_pred = y_pred[(month-1)*30:(month-1)*30+jours_afficher]

    moyenne_periode = np.mean(y_true)

    # Compute metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\nEvaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    ############-----PLOT-----################

    plt.figure(figsize=(12, 6), dpi=300)

    df_2020 = df[(df['Time'].dt.year == year_end)]

    test_dates_return = df_2020['Time'].reset_index(drop=True)
    test_dates_return = test_dates_return.dt.strftime('%d/%m/%Y')  # retourne une Series avec le bon format


    test_dates = df_2020['Time'].iloc[(month-1)*30:(month-1)*30+jours_afficher]

    y_true_plot = actuals_inv[(month-1)*30:(month-1)*30+jours_afficher].squeeze()
    y_pred_plot = predictions_inv[(month-1)*30:(month-1)*30+jours_afficher].squeeze()
    

    plt.plot(test_dates, y_true_plot, label='True',color ='red', linewidth=2.5)
    plt.plot(test_dates, y_pred_plot, label='Predicted',color='blue', linewidth=2.5, linestyle='--')

    plt.axhline(moyenne_annuelle, color='green', linestyle='--', linewidth=2, label="Annual mean")
    plt.axhline(moyenne_periode, color='orange', linestyle='--', linewidth=2, label="Period mean")

    plt.text(0.05, 0.95, f"R² = {r2:.3f}\n MSE = {mse:.3f}\n RMSE = {rmse:.3f}\n Number of days = {jours_afficher}", transform=plt.gca().transAxes,
         fontsize=12, color='black', verticalalignment='top')

    plt.title(f"{model.nom} Model Water Level Prediction at {model.nom_station} Station", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Water Level (m)", fontsize=14)

    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.gca().xaxis.set_major_locator(MaxNLocator(8))  # max 8 x-ticks
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)

    timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    plt.tight_layout()
    if save:
        plt.savefig( rf"C:\Users\roumi\OneDrive\Bureau\Projet_Stage\Data_plots\{model.nom}_plot\prediction_plot_{timestamp}_{model.nom_station}.png", dpi=300, bbox_inches='tight')
    plt.show()
    return y_true_return,y_pred_return,test_dates_return,y_true_plot,y_pred_plot,test_dates


def evaluate_models(models,loss_fn,val_loader,val_loader_graphe,scaler_y,df,edge_index,edge_weights):
    """
    Evalue plusieurs modèls succécivement
    """
    for model in models:
        if model.nom in ['CNN','Transformer']:
            evaluate_model(model=model,loss_fn=loss_fn,val_loader=val_loader,scaler_y=scaler_y,df=df)
        if model.nom in ['GNN','Model_Serie']:
            evaluate_model(model=model,loss_fn=loss_fn,val_loader=val_loader_graphe,scaler_y=scaler_y,df=df,edge_index=edge_index,edge_weights=edge_weights)
        
    print("Fin de l'évaluation des modèles")


print("Fin de complilation")

###########################################################################################################

def evaluate_combined(model_bas,seuil,model_haut,year_end,save,loss_fn,val_loader,scaler_y,df,month,jours_afficher):
    """
    Évalue un modèle selon son nom et trace la courbe de comparaison entre les valeurs prédites et réelles.

    Cette fonction permet d'afficher une courbe pour visualiser les performances du modèle.
    Args:
        model: Le modèle entraîné à évaluer.
        loss_fn: La fonction de perte utilisée.
        val_loader: Le DataLoader de validation.
        scaler_y: Le scaler utilisé pour les valeurs cibles (si applicable).
        df: Le DataFrame contenant les données d'origine.
        edge_index: La structure du graphe (indices des arêtes).
        edge_weights: Les poids des arêtes (si applicables).
    """
    # Collect predictions and true values
    predictions = []
    actuals = []
    ## Evaluate the model
    model_bas.eval()
    model_haut.eval()

    model = model_bas
    model.eval()

    with torch.no_grad():
        total_loss = 0
        for X_batch, y_batch in val_loader:
            output1 = model_haut(X_batch)
            output2 = model_bas(X_batch)
            output = (output1+output2)/2
            #output = model(X_batch)
            loss = loss_fn(output, y_batch)
            total_loss += loss.item()
            predictions.extend(output.squeeze().tolist())
            actuals.extend(y_batch.squeeze().tolist())
            if predictions[-1] > seuil:
                model = model_haut
            else:
                model = model_bas
       
    print(f"Test Loss: {total_loss / len(val_loader):.4f}")
    print("Nombre des predictions = ",len(predictions),"Nombre de actuels",len(actuals))
    
    # Convert predictions and actuals back to numpy arrays
    y_true = np.array(actuals)
    y_pred = np.array(predictions)

    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)

    predictions_inv = scaler_y.inverse_transform(predictions)
    actuals_inv = scaler_y.inverse_transform(actuals)

    y_true = actuals_inv.flatten()
    y_pred = predictions_inv.flatten()

    moyenne_annuelle = np.mean(y_true)

    y_true = y_true[(month-1)*30:(month-1)*30+jours_afficher]
    y_pred = y_pred[(month-1)*30:(month-1)*30+jours_afficher]

    moyenne_periode = np.mean(y_true)

    # Compute metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\nEvaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    ############-----PLOT-----################

    plt.figure(figsize=(12, 6), dpi=300)

    df_2020 = df[(df['Time'].dt.year == year_end)]

    test_dates = df_2020['Time'].iloc[(month-1)*30:(month-1)*30+jours_afficher]

    y_true_plot = actuals_inv[(month-1)*30:(month-1)*30+jours_afficher].squeeze()
    y_pred_plot = predictions_inv[(month-1)*30:(month-1)*30+jours_afficher].squeeze()
    

    plt.plot(test_dates, y_true_plot, label='True',color ='red', linewidth=2.5)
    plt.plot(test_dates, y_pred_plot, label='Predicted',color='blue', linewidth=2.5, linestyle='--')

    plt.axhline(moyenne_annuelle, color='green', linestyle='--', linewidth=2, label="Annual mean")
    plt.axhline(moyenne_periode, color='orange', linestyle='--', linewidth=2, label="Period mean")

    plt.text(0.05, 0.95, f"R² = {r2:.3f}\n MSE = {mse:.3f}\n RMSE = {rmse:.3f}\n Number of days = {jours_afficher}", transform=plt.gca().transAxes,
         fontsize=12, color='black', verticalalignment='top')

    plt.title(f"{model_bas.nom} and {model_haut.nom} Model Water Level Prediction at {model.nom_station} Station", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Water Level (m)", fontsize=14)

    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.gca().xaxis.set_major_locator(MaxNLocator(8))  # max 8 x-ticks
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)

    timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    plt.tight_layout()
    if save:
        plt.savefig( rf"C:\Users\roumi\OneDrive\Bureau\Projet_Stage\Data_plots\{model.nom}_plot\prediction_plot_{timestamp}_{model.nom_station}.png", dpi=300, bbox_inches='tight')
    plt.show()