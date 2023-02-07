import json
import pandas as pd
import numpy as np
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error

import stellargraph as sg
from stellargraph.mapper import HinSAGELinkGenerator
from stellargraph.layer import HinSAGE, link_regression
from tensorflow.keras import Model, optimizers, losses, metrics
from stellargraph import StellarGraph
from stellargraph import IndexedArray

import multiprocessing
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt

import tensorflow.keras.backend as K


def root_mean_square_error(s_true, s_pred):
    return K.sqrt(K.mean(K.pow(s_true - s_pred, 2)))

def accuracy(y_true, y_pred):
    correct = 0
    for i in range (len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1

    return correct/len(y_true)

def main():
    protein_interactions_train = pd.read_csv(
        'created_tables/setup2/seq_to_seq_train.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["source", "target"]  # set our own names for the columns
    )
    protein_interactions_train = protein_interactions_train.iloc[1:, :]

    protein_interactions_train_labels = pd.read_csv(
        'created_tables/setup2/seq_to_seq_train_withlabels.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["source", "target", "weight"]  # set our own names for the columns
    )
    protein_interactions_train_labels = protein_interactions_train_labels.iloc[1:, :]

    protein_interactions_test = pd.read_csv(
        'created_tables/setup2/seq_to_seq_test.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["source", "target"]  # set our own names for the columns
    )
    protein_interactions_test = protein_interactions_test.iloc[1:, :]

    protein_interactions_test_labels = pd.read_csv(
        'created_tables/setup2/seq_to_seq_test_withlabels.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["source", "target", "weight"]  # set our own names for the columns
    )
    protein_interactions_test_labels = protein_interactions_test_labels.iloc[1:, :]

    protein_interactions_train = protein_interactions_train.astype(int)
    protein_interactions_test = protein_interactions_test.astype(int)
    protein_interactions_train_labels = protein_interactions_train_labels.astype(int)
    protein_interactions_test_labels = protein_interactions_test_labels.astype(int)

    frames = [protein_interactions_train, protein_interactions_test]
    protein_interactions = pd.concat(frames, ignore_index=True)

    frames_train = [protein_interactions_train]
    protein_interactions_train = pd.concat(frames_train, ignore_index=True)

    frames_test = [protein_interactions_test]
    protein_interactions_test = pd.concat(frames_test, ignore_index=True)

    protein_word_content = pd.read_csv(
        'created_tables/setup2/seq_to_word.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
    )

    # protein_word_content = protein_word_content.iloc[1:, :]
    protein_word_content.drop(columns=protein_word_content.columns[0], axis=1, inplace=True)
    protein_word_content = protein_word_content.iloc[1:, :]

    for col in protein_word_content.columns:
        print(col)

    protein_word_content.columns = ['source', 'target']

    for col in protein_word_content.columns:
        print(col)

    protein_word_content = protein_word_content.astype(int)

    # Nodes (words-sequences)

    all_seq_id = pd.read_csv(
        'created_tables/setup2/all_seq_id.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
    )

    all_seq_id = all_seq_id.iloc[1:, :]
    all_seq_id.drop(columns=all_seq_id.columns[0], axis=1, inplace=True)
    all_seq_id.columns = ['index']
    all_seq_id.set_index("index", inplace=True)

    indexes = []
    for i in range(2741):
        if i < 2741:
            indexes.append(i)
    sequence_nodes = IndexedArray(index=indexes)

    sequence_nodes = np.identity(2741)
    sequence_nodes = pd.DataFrame(sequence_nodes)

    print('sequence nodes dataframe:', sequence_nodes)

    all_word_id = pd.read_csv(
        'created_tables/setup2/all_word_id.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
    )

    all_word_id = all_word_id.iloc[1:, :]
    all_word_id.drop(columns=all_word_id.columns[0], axis=1, inplace=True)
    all_word_id.columns = ['index']
    all_word_id.set_index("index", inplace=True)

    indexes = []
    for i in range(7502):
        if i > 2740:
            indexes.append(i)
    word_nodes = IndexedArray(index=indexes)

    word_nodes = np.identity(4761)
    word_nodes = pd.DataFrame(word_nodes)
    word_nodes.columns = indexes
    word_nodes.index = indexes

    print('word nodes dataframe:', word_nodes)

    protein_interactions_train_2 = protein_interactions_train[0:2]

    edges = [protein_interactions_train, protein_word_content]
    edges2 = [protein_interactions_train_2, protein_word_content]
    all_edges = pd.concat(edges, ignore_index=True)
    all_edges2 = pd.concat(edges2, ignore_index=True)

    edge_type = []
    for i in range(len(protein_interactions_train)):
        edge_type.append('i')
    for i in range(len(protein_word_content)):
        edge_type.append('c')

    all_edges['type'] = edge_type
    print(all_edges)

    edge_type2 = []
    for i in range(len(protein_interactions_train_2)):
        edge_type2.append('i')
    for i in range(len(protein_word_content)):
        edge_type2.append('c')

    all_edges2['type'] = edge_type2
    print(all_edges2)

    protein_train_graph = StellarGraph({"protein": sequence_nodes, "words": word_nodes}, edges=all_edges,
                                       edge_type_column="type")
    protein_train_graph_2 = StellarGraph({"protein": sequence_nodes, "word": word_nodes}, edges=all_edges2,
                                         edge_type_column="type")

    print(protein_train_graph.info())
    print(protein_train_graph_2.info())

    # ----------------------------------------------------------------------------------------------------
    # Training

    #########################################################################################################

    # Create own G_train, G_test graphs:
    G_test = protein_train_graph
    edge_ids_test = []
    edge_labels_test = []
    for row_index, row in protein_interactions_test_labels.iterrows():
        node1 = row['target']
        node2 = row['source']
        label = row['weight']
        edge = [node1, node2]
        edge_ids_test.append(edge)
        edge_labels_test.append(label)

    edge_ids_test = np.array(edge_ids_test)
    edge_labels_test = np.array(edge_labels_test)
    print('own edge ids test:', edge_ids_test)
    print('own edge labels test:', edge_labels_test)

    G_train = protein_train_graph_2
    edge_ids_train = []
    edge_labels_train = []
    for row_index, row in protein_interactions_train_labels.iterrows():
        if row_index >= 5:
            node1 = row['target']
            node2 = row['source']
            label = row['weight']
            edge = [node1, node2]
            edge_ids_train.append(edge)
            edge_labels_train.append(label)

    edge_ids_train = np.array(edge_ids_train)
    edge_labels_train = np.array(edge_labels_train)
    print('own edge ids test:', edge_ids_train)
    print('own edge labels test:', edge_labels_train)

    #########################################################################################################

    batch_size = 100
    epochs = 20
    # Use 70% of edges for training, the rest for testing:
    train_size = 0.7
    test_size = 0.3

    #dataset = datasets.MovieLens()
    #display(HTML(dataset.description))
    #G, edges_with_ratings = dataset.load()

    #print(G.info())

    #------------------------------------------------------------------------



    num_samples = [8, 4]

    generator = HinSAGELinkGenerator(
        protein_train_graph, batch_size, num_samples, head_node_types=["protein", "protein"]
    )

    train_gen = generator.flow(edge_ids_train, edge_labels_train, shuffle=True)
    test_gen = generator.flow(edge_ids_test, edge_labels_test)

    print(generator.schema.type_adjacency_list(generator.head_node_types, len(num_samples)))
    print(generator.schema.schema)

    hinsage_layer_sizes = [16, 16]
    assert len(hinsage_layer_sizes) == len(num_samples)

    hinsage = HinSAGE(
        layer_sizes=hinsage_layer_sizes, generator=generator, bias=True, dropout=0.0
    )

    # Expose input and output sockets of hinsage:
    x_inp, x_out = hinsage.in_out_tensors()

    # Final estimator layer
    score_prediction = link_regression(edge_embedding_method="concat")(x_out)


    model = Model(inputs=x_inp, outputs=score_prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=1e-2),
        loss=losses.mean_squared_error,
        metrics=[root_mean_square_error, metrics.mae],
    )

    print(model.summary())

    # Specify the number of workers to use for model training
    num_workers = 4

    test_metrics = model.evaluate(
        test_gen, verbose=1, use_multiprocessing=False, workers=num_workers
    )

    print("Untrained model's Test Evaluation:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=epochs,
        verbose=1,
        shuffle=False,
        use_multiprocessing=False,
        workers=num_workers,
    )

    sg.utils.plot_history(history)

    test_metrics = model.evaluate(
        test_gen, use_multiprocessing=False, workers=num_workers, verbose=1
    )

    print("Test Evaluation:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    y_true = edge_labels_test
    # Predict the rankings using the model:
    y_pred = model.predict(test_gen)
    # Mean baseline rankings = mean movie ranking:
    y_pred_baseline = np.full_like(y_pred, np.mean(y_true))

    print(y_true)
    print(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred_baseline))
    mae = mean_absolute_error(y_true, y_pred_baseline)
    acc = accuracy(y_true, y_pred_baseline)
    print("Mean Baseline Test set metrics:")
    print("\troot_mean_square_error = ", rmse)
    print("\tmean_absolute_error = ", mae)
    print("\tacc = ", acc)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    acc = accuracy(y_true, y_pred)
    print("\nModel Test set metrics:")
    print("\troot_mean_square_error = ", rmse)
    print("\tmean_absolute_error = ", mae)
    print("\tacc = ", acc)


if __name__ == '__main__':
    main()