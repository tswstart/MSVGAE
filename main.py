# -*- coding: utf-8 -*-
# @Author  : sw t
# @Time    : 2022/7/15 17:02
import sys
import os
import argparse
import torch

import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import umap
import hdbscan
from sklearn.cluster import KMeans
from tqdm.auto import tqdm
import torch_geometric.transforms as T
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI

from model.MSVGAE import MSVGAE
from model.MSVGAE_Encoder import GAT_Encoder, GCN_Encoder




def preprocess(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


def load_X_file(path):
    pre_path, filename = os.path.split(path)
    dataset_name, ext = os.path.splitext(filename)
    args.name = dataset_name
    if path[-5:] == '.h5ad':
        adata = anndata.read_h5ad(path)
    elif path[-4:] == '.csv':
        adata = anndata.AnnData(pd.read_csv(path, index_col=0))
    return adata


def load_Y_file(path):
    if path[-5:] == '.h5ad':
        gd_labels = anndata.read_h5ad(path).obs['x'].values
    if path[-4:] == '.csv':
        gd_labels = pd.read_csv(path, index_col=0)['x'].values
    return gd_labels


def prepare_data(args):
    # raw counts
    print('Preparing data...')
    adata = load_X_file(args['X_path'])
    gd_labels = load_Y_file(args['Y_path'])

    print(f'raw data shape: {adata.shape}')
    print(f'raw labels shape: {gd_labels.shape}')

    # for raw data, whether deal with or not
    adata_pp = adata.copy()
    if args['preprocess']:
        print('Applying preprocessing...')
        preprocess(adata_pp)
    else:
        print('Applying log-normalisation...')
        sc.pp.log1p(adata_pp, copy=False)

    adata_hvg = adata_pp.copy()
    adata_ghvg = adata_pp.copy()
    sc.pp.highly_variable_genes(adata_hvg, n_top_genes=args['hvg'], inplace=True, flavor='seurat')
    sc.pp.highly_variable_genes(adata_ghvg, n_top_genes=args['ghvg'], inplace=True, flavor='seurat')

    adata_hvg = adata_hvg[:, adata_hvg.var['highly_variable'].values]
    adata_ghvg = adata_ghvg[:, adata_ghvg.var['highly_variable'].values]
    X_hvg = adata_hvg.X
    X_ghvg = adata_ghvg.X

    print(f'HVG adata shape: {adata_hvg.shape}')
    print(f'GHVG adata shape: {adata_ghvg.shape}')

    return adata_hvg, adata_ghvg, X_hvg, X_ghvg, gd_labels


def load_hvg(hvg_path):
    adata = load_X_file(hvg_path)
    return adata


def load_graph(edge_path):
    edgelist = []
    with open(edge_path, 'r') as edge_file:
        edgelist = [(int(item.split()[0]), int(item.split()[1])) for item in edge_file.readlines()]
    return edgelist


def correlation(data_numpy, k, corr_type='pearson'):
    df = pd.DataFrame(data_numpy.T)
    corr = df.corr(method=corr_type)
    nlargest = k
    order = np.argsort(-corr.values, axis=1)[:, :nlargest]
    neighbors = np.delete(order, 0, 1)

    return corr, neighbors


def prepare_graphs(adata_khvg, X_khvg, args):
    if args['graph_type'] == 'KNN':
        print('Computing KNN graph by scanpy...')
        # use package scanpy to compute knn graph
        distances_csr_matrix = \
            sc.pp.neighbors(adata_khvg, n_neighbors=args['k'] + 1, n_pcs=args['graph_n_pcs'], knn=True, copy=True).obsp[
                'distances']
        # ndarray
        distances = distances_csr_matrix.A
        # resize
        neighbors = np.resize(distances_csr_matrix.indices, new_shape=(distances.shape[0], args['k']))

    elif args['graph_type'] == 'PKNN':
        print('Computing PKNN graph...')
        distances, neighbors = correlation(data_numpy=X_khvg, k=args['k'] + 1)

    if args['graph_distance_cutoff_num_stds']:
        cutoff = np.mean(np.nonzero(distances), axis=None) + float(args['graph_distance_cutoff_num_stds']) * np.std(
            np.nonzero(distances), axis=None)
    # shape: 2 * (the number of edge)
    edgelist = []
    for i in range(neighbors.shape[0]):
        for j in range(neighbors.shape[1]):
            pair = (str(i), str(neighbors[i][j]))
            if args['graph_distance_cutoff_num_stds']:
                distance = distances[i][j]
                if distance < cutoff:
                    if i != neighbors[i][j]:
                        edgelist.append(pair)
            else:
                if i != neighbors[i][j]:
                    edgelist.append(pair)

    print(f'The graph has {len(edgelist)} edges.')

    if args['save_graph']:
        Path(args['save_path']).mkdir(parents=True, exist_ok=True)

        num_hvg = X_khvg.shape[1]
        k_file = args['k']
        if args['graph_type'] == 'KNN':
            graph_name = 'Scanpy'
        elif args['graph_type'] == 'PKNN':
            graph_name = 'Pearson'

        if args['name']:
            filename = f'{args["name"]}_{graph_name}_KNN_K{k_file}_gHVG_{num_hvg}.txt'
        else:
            filename = f'{graph_name}_KNN_K{k_file}_gHVG_{num_hvg}.txt'

        if args['graph_n_pcs']:
            filename = filename.split('.')[0] + f'_d_{args["graph_n_pcs"]}.txt'
        if args['graph_distance_cutoff_num_stds']:
            filename = filename.split('.')[0] + '_cutoff_{:.4f}.txt'.format(cutoff)

        final_path = os.path.join(args['save_path'], filename)
        print(f'Saving graph to {final_path}...')
        with open(final_path, 'w') as f:
            edges = [' '.join(e) + '\n' for e in edgelist]
            f.writelines(edges)

    return edgelist


def train(model, optimizer, train_data, device):
    # set training mode
    model = model.train()
    # initialize epoch_loss
    epoch_loss = 0.0
    # put data into gpu
    x, edge_index = train_data.x.to(torch.float).to(device), train_data.edge_index.to(torch.long).to(device)

    optimizer.zero_grad()

    # encode
    z = model.encode(x, edge_index)
    # inner decoder recon loss
    reconstruction_loss = model.recon_loss(z, train_data.pos_edge_label_index)
    # kl_loss
    loss = reconstruction_loss + (1 / train_data.num_nodes) * model.kl_loss()
    # liner decoder loss
    decoder_loss = 0.0
    reconstructed_features = model.liner_decoder(z)
    decoder_loss = torch.nn.functional.mse_loss(reconstructed_features, x) * 10
    loss += decoder_loss
    # backpropagation
    loss.backward()
    # update grad
    optimizer.step()

    epoch_loss += loss.item()
    return epoch_loss, decoder_loss  # liner decoder loss


def compute_metrics(y_true, y_pred):
    metrics = {}
    metrics["ARI"] = ARI(y_true, y_pred)
    metrics["NMI"] = NMI(y_true, y_pred)

    return metrics


@torch.no_grad()
def inference(model, device, data, args):
    model = model.eval()
    # firstly encode
    z = model.encode(data.x.to(torch.float).to(device), data.edge_index.to(torch.long).to(device))
    # hdbscan
    if args['hdbscan']:
        umap_reducer = umap.UMAP()
        u = umap_reducer.fit_transform(z.cpu().detach().numpy())
        cl_sizes = [10, 25, 50, 100]
        # cl_sizes = range(10, 31)
        min_samples = [5, 10, 25, 50]
        # min_samples = range(10, 31)
        hdbscan_dict = {}
        ari_dict = {}
        for cl_size in cl_sizes:
            for min_sample in min_samples:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=cl_size, min_samples=min_sample)
                clusterer.fit(u)
                ari_dict[(cl_size, min_sample)] = compute_metrics(data.y, clusterer.labels_)
                hdbscan_dict[(cl_size, min_sample)] = clusterer.labels_
        max_tuple = max(ari_dict, key=lambda x: ari_dict[x]['ARI'])
        return ari_dict[max_tuple], hdbscan_dict[max_tuple], z

    # kmeans
    if args['kmeans']:
        num_cluster = len(set(data.y))
        pd_labels = KMeans(n_clusters=num_cluster).fit(z.cpu().detach().numpy()).labels_
        # compute metrics
        eval_supervised_metrics = compute_metrics(data.y, pd_labels)
        return eval_supervised_metrics, pd_labels, z


def setup(args, device):
    # prepare data of X
    # return the high variable genes corresponding to adata and adata.X
    adata_hvg, adata_khvg, X_hvg, X_khvg, gt_labels = prepare_data(args)

    # prepare graph
    if not args['graph_path']:
        edgelist = prepare_graphs(adata_khvg, X_khvg, args)  # khvg用来构建图
    else:
        edgelist = load_graph(args['graph_path'])

    num_nodes = X_hvg.shape[0]
    print(f'Number of nodes in graph: {num_nodes}.')
    edge_index = np.array(edgelist).astype(int).T
    edge_index = to_undirected(torch.from_numpy(edge_index).to(torch.long), num_nodes)
    # MinMaxScaler
    scaler = MinMaxScaler()
    scaled_x = torch.from_numpy(scaler.fit_transform(X_hvg.toarray()))
    # data encapsulation
    data_obj = Data(edge_index=edge_index, x=scaled_x)
    data_obj.num_nodes = X_hvg.shape[0]
    data_obj.y = gt_labels

    data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = None

    if (args['load_model_path'] is not None):
        # PyTorch Geometric does not allow 0 training samples (all test), so we need to store all test data as 'training'.
        val_split = 0.0
    else:
        val_split = args['val_split']

    # Can set validation ratio
    add_negative_train_samples = args['load_model_path'] is not None

    transform = T.RandomLinkSplit(num_val=val_split, num_test=0.0, is_undirected=True,
                                  add_negative_train_samples=add_negative_train_samples, split_labels=True)
    train_data, val_data, test_data = transform(data_obj)

    num_features = data_obj.num_features
    # initialize
    num_heads = {}
    num_heads['first'] = args['num_heads'][0]
    num_heads['second'] = args['num_heads'][1]
    num_heads['mean'] = args['num_heads'][2]
    num_heads['std'] = args['num_heads'][3]
    # two encoder
    if args['GAT']:
        encoder_1 = GAT_Encoder(
            in_channels=num_features,
            num_heads=num_heads,
            hidden_dims=args['hidden_dims'],
            dropout=args['dropout'],
            latent_dim=args['latent_dim']
        )
        encoder_2 = GAT_Encoder(
            in_channels=num_features,
            num_heads=num_heads,
            hidden_dims=args['hidden_dims'],
            dropout=args['dropout'],
            latent_dim=args['latent_dim']
        )
    if args['GCN']:
        encoder_1 = GCN_Encoder(
            in_channels=num_features,
            hidden_dims=args['hidden_dims'],
            latent_dim=args['latent_dim']
        )
        encoder_2 = GCN_Encoder(
            in_channels=num_features,
            hidden_dims=args['hidden_dims'],
            latent_dim=args['latent_dim']
        )


    model = MSVGAE(encoder_gat1=encoder_1, encoder_gat2=encoder_2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    model = model.to(device)

    return model, optimizer, train_data, val_data


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # convert vars to dict
    args = vars(args)
    hidden_dims = args['hidden_dims']
    num_heads = args['num_heads']
    dropout = args['dropout']

    # initialize
    model, optimizer, train_data, val_data = setup(args, device=device)
    if torch.cuda.is_available():
        print(f'\nCUDA available, using {torch.cuda.get_device_name(device)}.')
    print('Neural model details: \n')
    print(model)

    print(f'Using {args["latent_dim"]} latent dimensions.')

    print(f'Number of train edges = {train_data.edge_index.shape[1]}.\n')

    if args['val_split']:
        print(
            f'Using validation split of {args["val_split"]}, number of validation edges = {val_data.pos_edge_label_index.shape[1]}.')

    # Train/val/test code
    print('Training model...')
    best_ari, best_eval_supervised_metrics, best_pd_labels, best_embeddings = -1, None, None, None
    for epoch in tqdm(range(1, args['epochs'] + 1)):  # decoder_loss
        epoch_loss, decoder_loss = train(model, optimizer, train_data, device=device)
        print('Epoch {:03d} -- Total epoch loss: {:.4f} -- NN decoder epoch loss: {:.4f}'.format(epoch, epoch_loss,
                                                                                                 decoder_loss))
        # inference log & supervised metrics
        if epoch % 10 == 0 or epoch == args['epochs']:
            metrics, pred_labels, z_nodes = inference(model=model, device=device, data=val_data, args=args)
            print('Inference ARI:{} NMI:{}'.format(metrics['ARI'], metrics['NMI']))
            if metrics['ARI'] > best_ari:
                best_ari = metrics['ARI']
                best_eval_supervised_metrics = metrics
                best_pd_labels = pred_labels
                best_embeddings = z_nodes
            with open(os.path.join(args['save_path'], 'log_MSVGAE_{}.txt'.format(args['name'])), "a") as f:
                f.writelines("{}\teval\t{}\n".format(epoch, metrics))

    # Save node embeddings
    node_embeddings = []
    # detach into cpu
    node_embeddings.append(best_embeddings.cpu().detach().numpy())
    node_embeddings = np.array(node_embeddings)
    node_embeddings = node_embeddings.squeeze()
    # save embedding
    Path(args['save_path']).mkdir(parents=True, exist_ok=True)
    if args['name']:
        filename = f'{args["name"]}_MSVGAE_node_embeddings.npy'
    else:
        filename = 'MSVGAE_node_embeddings.npy'

    node_filepath = os.path.join(args['save_path'], filename)
    np.save(node_filepath, node_embeddings)

    # save pd_label
    if best_pd_labels is not None:
        pd_labels_df = pd.DataFrame(best_pd_labels, columns=['pd_label'])
        pd_labels_df.to_csv(os.path.join(args['save_path'], "pd_label_MSVGAE_{}.csv".format(args['name'])))

    # save best metrics
    best_metrics = best_eval_supervised_metrics
    txt_path = os.path.join(args['save_path'], "metric_MSVGAE.txt")
    f = open(txt_path, "a")
    record_string = args['name']
    for key in best_metrics.keys():
        record_string += " {}".format(best_metrics[key])
    record_string += "\n"
    f.write(record_string)
    f.close()

    # Save model
    if args['save_model']:
        if args['name']:
            filename = f'{args["name"]}_MSVGAE_model.pt'
        else:
            filename = 'MSVGAE_model.pt'
        model_filepath = os.path.join(args['save_path'], filename)
        torch.save(model.state_dict(), model_filepath)


if __name__ == '__main__':
    # create parser
    parser = argparse.ArgumentParser(prog='MSVGAE',
                                     description='Using multi-encoder semi-implicit graph variational autoencoder to analyze single-cell RNA sequence data.')

    parser.add_argument('--X_path', help='Input gene expression matrix file path.It is only .h5ad and .csv')
    parser.add_argument('--Y_path', help='')
    parser.add_argument('--hvg', type=int, help='Number of highly variable genes.', default=1024)
    parser.add_argument('--ghvg', type=int, help='Number of highly variable genes used to create knn graph.',
                        default=1024)
    parser.add_argument('--GAT', action='store_true', help='use GAT', default=False)
    parser.add_argument('--GCN', action='store_true', help='use GCN', default=False)
    parser.add_argument('--graph_type', choices=['KNN', 'PKNN'], default='KNN')
    parser.add_argument('--k', type=int, help='Number of neighbors for graph', default=5)
    parser.add_argument('--graph_n_pcs', type=int, default=50)
    parser.add_argument('--graph_distance_cutoff_num_stds', type=float, default=0.0)
    parser.add_argument('--save_graph', action='store_true', default=False,
                        help='Save the generated graph to the output path specified by --save_path.')
    # for raw_counts,whether preprocess or not
    parser.add_argument('--preprocess', action='store_true', help='Preprocess recipe for raw counts.')
    parser.add_argument('--graph_path', help='Graph specified as an edge list (one edge per line, nodes separated by whitespace, not comma), if not using command line options to generate it.')
    parser.add_argument('--num_heads', help='Number of attention heads for each layer. Input is a list that must match the total number of layers = num_hidden_layers + 2 in length.',
                        type=int, nargs='*', default=[10, 10, 10, 10])
    parser.add_argument('--hidden_dims', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--dropout', type=float, nargs='*', default=(0.4, 0.4))
    parser.add_argument('--latent_dim', help='Latent dimension (output dimension for node embeddings).', default=100,
                        type=int)
    parser.add_argument('--hdbscan', action='store_true', default=False)
    parser.add_argument('--kmeans', action='store_true', default=False)
    parser.add_argument('--lr', help='Learning rate for Adam.', default=0.001, type=float)
    parser.add_argument('--epochs', help='Number of training epochs.', default=100, type=int)
    parser.add_argument('--val_split', help='Validation split e.g. 0.1.', default=0.0, type=float)
    parser.add_argument('--name', help='Name used for the written output files.', type=str, default='')
    parser.add_argument('--save_path', type=str, default='./results/')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--load_model_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args)
