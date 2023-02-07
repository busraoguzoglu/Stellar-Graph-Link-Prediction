import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import FullBatchLinkGenerator
from stellargraph.layer import GCN, LinkEmbedding
import pandas as pd
import numpy as np
from stellargraph import StellarGraph

import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, HinSAGE, link_classification

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection

from stellargraph import globalvar
from stellargraph import datasets
from IPython.display import display, HTML

def main():

    protein_interactions_train = pd.read_csv(
        'created_tables/setup1/seq_to_seq_train.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["target", "source"]  # set our own names for the columns
    )
    protein_interactions_train = protein_interactions_train.iloc[1:, :]

    protein_interactions_train_labels = pd.read_csv(
        'created_tables/setup1/seq_to_seq_train_withlabels.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["target", "source", "weight"]  # set our own names for the columns
    )
    protein_interactions_train_labels = protein_interactions_train_labels.iloc[1:, :]

    protein_interactions_test = pd.read_csv(
        'created_tables/setup1/seq_to_seq_test.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["target", "source"]  # set our own names for the columns
    )
    protein_interactions_test = protein_interactions_test.iloc[1:, :]

    protein_interactions_test_labels = pd.read_csv(
        'created_tables/setup1/seq_to_seq_test_withlabels.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["target", "source", "weight"]  # set our own names for the columns
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

    # Shuffle
    #protein_interactions = protein_interactions.sample(frac=1)

    protein_word_content = pd.read_csv(
        'created_tables/setup1/seq_to_word.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
    )

    #protein_word_content = protein_word_content.iloc[1:, :]
    #protein_word_content = protein_word_content.astype(int)

    # Words as features

    protein_word_content = protein_word_content.rename(columns={0: 'id'})
    protein_word_content["id"] = protein_word_content.index
    protein_word_content = protein_word_content.set_index("id")

    protein_word_content = protein_word_content.iloc[1:, :]

    #protein_interactions = protein_interactions.drop(['weight'], axis=1)
    print('protein_word_content: ', protein_word_content)

    protein_interactions_train_2 = protein_interactions_train[0:5]

    print(protein_interactions)
    protein_graph = StellarGraph({"protein": protein_word_content}, {"interacts": protein_interactions})

    protein_train_graph = StellarGraph({"protein": protein_word_content}, {"interacts": protein_interactions_train})
    protein_train_graph_2 = StellarGraph({"protein": protein_word_content}, {"interacts": protein_interactions_train_2})
    protein_test_graph = StellarGraph({"protein": protein_word_content}, {"interacts": protein_interactions_test})

    print(protein_graph.info())

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

    batch_size = 20
    epochs = 20

    num_samples = [40, 20]

    train_gen = GraphSAGELinkGenerator(G_train, batch_size, num_samples)
    train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)

    test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples)
    test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

    layer_sizes = [100, 100]
    graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=0.2
    )

    # Build the model and expose input and output sockets of graphsage model
    # for link prediction
    x_inp, x_out = graphsage.in_out_tensors()

    prediction = link_classification(
        output_dim=1, output_act="relu", edge_embedding_method="ip"
    )(x_out)

    model = keras.Model(inputs=x_inp, outputs=prediction)

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=["acc"],
    )

    init_train_metrics = model.evaluate(train_flow)
    init_test_metrics = model.evaluate(test_flow)

    print("\nTrain Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    history = model.fit(train_flow, epochs=epochs, validation_data=test_flow, verbose=2)

    train_metrics = model.evaluate(train_flow)
    test_metrics = model.evaluate(test_flow)

    print("\nTrain Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

if __name__ == '__main__':
    main()

