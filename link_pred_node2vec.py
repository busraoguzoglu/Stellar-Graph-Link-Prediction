import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import FullBatchLinkGenerator
from stellargraph.layer import GCN, LinkEmbedding
import pandas as pd
import numpy as np
from stellargraph import StellarGraph
from stellargraph import IndexedArray

from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import multiprocessing

def node2vec_embedding(graph, name):

    p = 1.0
    q = 1.0
    dimensions = 128
    num_walks = 10
    walk_length = 80
    window_size = 10
    num_iter = 1
    workers = multiprocessing.cpu_count()


    rw = BiasedRandomWalk(graph)
    walks = rw.run(graph.nodes(), n=num_walks, length=walk_length, p=p, q=q)
    print(f"Number of random walks for '{name}': {len(walks)}")

    model = Word2Vec(
        walks,
        vector_size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,
        workers=workers,
        epochs=num_iter,
    )

    def get_embedding(u):
        return model.wv[u]

    return get_embedding

# 1. link embeddings
def link_examples_to_features(link_examples, transform_node, binary_operator):
    return [
        binary_operator(transform_node(src), transform_node(dst))
        for src, dst in link_examples
    ]


# 2. training classifier
def train_link_prediction_model(
    link_examples, link_labels, get_embedding, binary_operator
):
    clf = link_prediction_classifier()
    link_features = link_examples_to_features(
        link_examples, get_embedding, binary_operator
    )
    clf.fit(link_features, link_labels)
    return clf


def link_prediction_classifier(max_iter=2000):
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])


# 3. and 4. evaluate classifier
def evaluate_link_prediction_model(
    clf, link_examples_test, link_labels_test, get_embedding, binary_operator
):
    link_features_test = link_examples_to_features(
        link_examples_test, get_embedding, binary_operator
    )
    score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
    return score


def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])

def operator_hadamard(u, v):
    return u * v


def operator_l1(u, v):
    return np.abs(u - v)


def operator_l2(u, v):
    return (u - v) ** 2


def operator_avg(u, v):
    return (u + v) / 2.0


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

    #protein_word_content = protein_word_content.iloc[1:, :]
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

    protein_interactions_train_2 = protein_interactions_train[0:5]

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

    protein_train_graph = StellarGraph({"protein": sequence_nodes, "words": word_nodes}, edges = all_edges, edge_type_column="type")
    protein_train_graph_2 = StellarGraph({"protein": sequence_nodes, "word": word_nodes}, edges = all_edges2, edge_type_column="type")

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

    # Training:
    embedding_train = node2vec_embedding(G_train, "Train Graph")
    binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]

    results = []

    # Different operators:
    ########################################################################
    # 1.
    clf = train_link_prediction_model(
        edge_ids_train, edge_labels_train, embedding_train, operator_l2
    )
    score = evaluate_link_prediction_model(
        clf,
        edge_ids_test,
        edge_labels_test,
        embedding_train,
        operator_l2,
    )

    results.append({
        "classifier": clf,
        "binary_operator": operator_l2,
        "score": score,
    })

    # 2.
    clf = train_link_prediction_model(
        edge_ids_train, edge_labels_train, embedding_train, operator_l1
    )
    score = evaluate_link_prediction_model(
        clf,
        edge_ids_test,
        edge_labels_test,
        embedding_train,
        operator_l1,
    )

    results.append({
        "classifier": clf,
        "binary_operator": operator_l1,
        "score": score,
    })
    #######################################################################

    best_result = max(results, key=lambda result: result["score"])

    print(f"Best result from '{best_result['binary_operator'].__name__}'")

    pd.DataFrame(
        [(result["binary_operator"].__name__, result["score"]) for result in results],
        columns=("name", "ROC AUC score"),
    ).set_index("name")

    embedding_test = node2vec_embedding(G_test, "Test Graph")

    test_score = evaluate_link_prediction_model(
        best_result["classifier"],
        edge_ids_test,
        edge_labels_test,
        embedding_test,
        best_result["binary_operator"],
    )
    print(
        f"ROC AUC score on test set using '{best_result['binary_operator'].__name__}': {test_score}"
    )

if __name__ == '__main__':
    main()

