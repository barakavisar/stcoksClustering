from sklearn.cluster import k_means
import pandas as pd
import numpy as np
import os

def run_cluster_algo(symbols, algo_directory, n_clusters):

    # gather past close prices (time series) of 200 days from all symbols
    processed_symbols = []
    for i in range(len(symbols)):
        symbol_data = pd.read_csv(algo_directory + '/' + symbols[i] + '.csv')

        try:
            symbol_data = np.array(symbol_data['Close'], dtype=float)[-200:]
        except:
            continue
        processed_symbols.append(symbols[i])

        if i == 0:
            data = symbol_data
        else:
            data = np.vstack((data, symbol_data))
  
    model = 'k_means'

    num_stocks = data.shape[0]
    len_days = data.shape[1]

    print('n stocks', num_stocks)
    data = data.T

    # define dataset as dataframe
    X = pd.DataFrame(data, dtype=float)

    # computes distances matrix
    corralations_matrix = X.corr("pearson")

    print('upper left corner of correlation matrix')
    print(corralations_matrix.iloc[:5, :5])

    # In some cases setting the matrix diagonal to zero
    #for rmx in range(distances_matrix.shape[0]):
    #   for rmy in range(distances_matrix.shape[0]):
    #       if rmx == rmy:
    #           distances_matrix[rmx, rmy] = 0

    # fit model and predict clusters
    if model == 'k_means':
        clusters = k_means(corralations_matrix, n_clusters)

    # print(len(clusters[1]))
    print('first 15 stocks clusters ids')
    print(clusters[1][:15])

    clusters_sizes = []
    clusters = np.array(clusters[1], dtype=int)
    for k in range(n_clusters):
        curr_indexes = np.where(clusters == k)[0]
        clusters_sizes.append(len(curr_indexes))

    print('clusters sizes')
    print(clusters_sizes)

    # write summary result file
    result_path = algo_directory + '/' + str(model)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    clusters_ids = np.array(range(n_clusters), dtype=int)
    clusters_ids = pd.DataFrame(clusters_ids)
    clusters_sizes = pd.DataFrame(clusters_sizes)
    result = pd.concat((clusters_ids, clusters_sizes), axis=1)
    result.columns = ['cluster_id', 'cluster_size']
    result.to_csv(result_path + '/clustering_summary_result.csv')

    # write detailed result
    clusters = pd.DataFrame(clusters)
    stocks = pd.DataFrame(processed_symbols)
    detailed_result = pd.concat((stocks, clusters), axis=1)
    detailed_result.columns = ['symbol', 'cluster_id']
    detailed_result.to_csv(result_path + '/clustering_detailed_result.csv')


def initiate():

    # define path to symbols, algo directory and number of clusters
    symbols_path = '.../SP500/symbols.csv'
    algo_directory = '......./SP500'
    n_clusters = 11
    
    symbols = pd.read_csv(symbols_path)
    symbols = np.array(symbols['Symbol'], dtype=str)

    run_cluster_algo(symbols, algo_directory, n_clusters)

initiate()


