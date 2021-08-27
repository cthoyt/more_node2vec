# -*- coding: utf-8 -*-

"""Main code."""

from __future__ import annotations

import csv
import datetime
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING, Type, Union

import click
import networkx as nx
import numpy as np
import pandas as pd
from nodevectors import Node2Vec
from sklearn.base import BaseEstimator

if TYPE_CHECKING:
    from gensim.models.keyedvectors import Word2VecKeyedVectors

__all__ = [
    "echo",
    "Model",
    "load_tabbed_word2vec_format",
    "save_tabbed_word2vec_format",
    "reader",
    "writer",
    # Graph utils
    "get_undirected_graph_from_df",
    "process_graph",
    "fit_model",
]


@dataclass
class Model:
    """Wraps keyed vectors for downstream model building."""

    vector_name: ClassVar[str] = "embeddings.tsv.gz"
    vocab_name: ClassVar[str] = "vocab.tsv.gz"

    wv: "Word2VecKeyedVectors"

    @property
    def vocab(self):
        """Access the keyed vector vocabulary."""
        return self.wv.vocab

    @property
    def vectors(self) -> np.ndarray:
        """Access the vectors."""
        return self.wv.vectors

    def get_labels(self, indices: Union[str, list[str], np.ndarray]) -> np.ndarray:
        """Get the label for the given indices."""
        if isinstance(indices, int):
            return self.wv.index2entity[indices]
        return np.vectorize(self.wv.index2entity.__getitem__)(indices)

    def __getitem__(self, index: Union[str, list[str]]):  # noqa:D105
        return self.wv.__getitem__(index)

    def as_dict(self) -> dict[str, np.ndarray]:
        """Get the vectors as a dictionary keyed by the vocabulary."""
        return dict(zip(self.wv.vocab, self.wv.vectors))

    @staticmethod
    def from_node2vec(node2vec: Node2Vec) -> Model:
        """Get the model from a :class:`nodevectors.Node2Vec` instance."""
        return Model(wv=node2vec.model.wv)

    def save(self, directory: Union[str, Path]) -> None:
        """Save the model to the given directory."""
        if isinstance(directory, str):
            directory = Path(directory)
        directory = directory.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        save_tabbed_word2vec_format(
            wv=self.wv,
            vectors_path=directory / self.vector_name,
            vocab_path=directory / self.vocab_name,
        )

    @classmethod
    def load(cls, directory: Union[str, Path]) -> Model:
        """Load the model from the given directory."""
        if isinstance(directory, str):
            directory = Path(directory)
        directory = directory.resolve()
        wv = load_tabbed_word2vec_format(
            vectors_path=directory / cls.vector_name,
            vocab_path=directory / cls.vocab_name,
        )
        return Model(wv=wv)

    def reduce(
        self,
        n_components: Union[None, int, float] = 2,
        reducer: Union[None, str, Type[BaseEstimator]] = None,
    ) -> tuple[BaseEstimator, np.ndarray]:
        """Return a reduced version of the vectors in a numpy array and the scikit-learn estimator."""
        estimator = get_reducer(n_components, reducer)
        return estimator, estimator.fit_transform(self.wv.vectors)

    def reduce_df(
        self,
        n_components: Union[None, int, float] = 2,
        reducer: Union[None, str, Type[BaseEstimator]] = None,
    ) -> tuple[BaseEstimator, pd.DataFrame]:
        """Return a reduced version of the vectors in a dataframe and the scikit-learn estimator."""
        estimator, x = self.reduce(n_components=n_components, reducer=reducer)
        return estimator, pd.DataFrame(x, index=self.wv.vocab)


def load_tabbed_word2vec_format(
    vectors_path: Union[str, Path], vocab_path: Union[str, Path], dtype=np.float64
) -> "Word2VecKeyedVectors":
    """Load the input-hidden weight matrix from the original C word2vec-tool format."""
    from gensim.models.keyedvectors import Vocab, Word2VecKeyedVectors

    with reader(vectors_path) as vector_reader, reader(vocab_path) as vocab_reader:
        vector_header = next(vector_reader)
        try:
            vocab_size, vector_size = (int(x) for x in vector_header)
        except ValueError:
            # throws for invalid file format
            raise ValueError(f"header row should be two integers: {vector_header}")

        result = Word2VecKeyedVectors(vector_size)
        result.vector_size = vector_size
        result.vectors = np.zeros((vocab_size, vector_size), dtype=dtype)

        it = enumerate(zip(vector_reader, vocab_reader))
        for index, ((_, *weights), vocab_line) in it:
            try:
                word, count = vocab_line
            except ValueError:
                raise ValueError(f"Error on vocab line {index}: {vocab_line}")

            if len(weights) != vector_size:
                raise ValueError(
                    f"[line {index + 1}] vector of wrong dimension"
                    f" (got {len(weights)}, should be {vector_size})",
                )
            weights = [dtype(weight) for weight in weights]

            # always increasing
            result.vocab[word] = Vocab(index=index, count=count)
            result.vectors[index] = weights
            result.index2word.append(word)

    return result


def save_tabbed_word2vec_format(
    *,
    wv: "Word2VecKeyedVectors",
    vectors_path: Union[str, Path],
    vocab_path: Union[str, Path],
) -> None:
    """Save the word2vec format."""
    sorted_vocab_items = sorted(wv.vocab.items(), key=lambda item: item[1].count, reverse=True)
    total_vec = len(wv.vocab)
    vectors = wv.vectors
    vector_size = vectors.shape[1]
    if total_vec != vectors.shape[0]:
        raise ValueError

    with writer(vectors_path) as vectors_writer, writer(vocab_path) as vocab_writer:
        vectors_writer.writerow((total_vec, vector_size))
        for word, vocab_ in sorted_vocab_items:
            # Write to vocab file
            vocab_writer.writerow((word, vocab_.count))
            # Write to vectors file
            vector_row = word, *(repr(val) for val in vectors[vocab_.index])
            vectors_writer.writerow(vector_row)


@contextmanager
def writer(path: Union[str, Path]):
    """Open a CSV writer context manager."""
    opener, kwargs = _get_writer(path)
    with opener(path, **kwargs) as file:
        yield csv.writer(file, delimiter="\t", quoting=csv.QUOTE_MINIMAL)


def _get_writer(path: Union[str, Path]):
    """Get the file opener."""
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == ".gz":
        return gzip.open, dict(mode="wt")
    return open, {"mode": "w"}


@contextmanager
def reader(path: Union[str, Path]):
    """Open a CSV reader context manager."""
    opener, kwargs = _get_reader(path)
    with opener(path, **kwargs) as file:
        yield csv.reader(file, delimiter="\t", quoting=csv.QUOTE_MINIMAL)


def _get_reader(path: Union[str, Path]):
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == ".gz":
        return gzip.open, {"mode": "rt"}
    return open, {"mode": "r"}


def echo(*s, sep=" ", **kwargs):
    """Write with :func:`click.secho` preceed by the time."""
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    log_str = sep.join(str(x) for x in s)
    click.echo(click.style(f"[{time_str}] ", fg="blue") + click.style(log_str, **kwargs))


def get_reducer(n_components, model: Union[None, str, Type[BaseEstimator]], **kwargs):
    """Get a reducer model instance."""
    reducer_cls, reducer_kwargs = get_reducer_cls(model, **kwargs)
    return reducer_cls(n_components, **reducer_kwargs)


def get_reducer_cls(model: Union[None, str, Type[BaseEstimator]], **kwargs):
    """Get the model class by name and default kwargs.

    :param model: The name of the model. Can choose from: PCA, KPCA, GRP,
        SRP, TSNE, LLE, ISOMAP, MDS, or SE.
    :param kwargs: Keyword arguments that will get passed through and modified based on the chosen model.
    :return: A pair of a reducer class from :mod:`sklearn` and the modified kwargs.

    :raises ValueError: if invalid model name is passed
    """
    if isinstance(model, type):
        return model, {}
    if model is None or model.upper() == "PCA":
        from sklearn.decomposition import PCA as Reducer  # noqa:N811
    elif model.upper() == "KPCA":
        kwargs.setdefault("kernel", "rbf")
        from sklearn.decomposition import KernelPCA as Reducer
    elif model.upper() == "GRP":
        from sklearn.random_projection import GaussianRandomProjection as Reducer
    elif model.upper() == "SRP":
        from sklearn.random_projection import SparseRandomProjection as Reducer
    elif model.upper() in {"T-SNE", "TSNE"}:
        from sklearn.manifold import TSNE as Reducer  # noqa:N811
    elif model.upper() in {"LLE", "LOCALLYLINEAREMBEDDING"}:
        from sklearn.manifold import LocallyLinearEmbedding as Reducer
    elif model.upper() == "ISOMAP":
        from sklearn.manifold import Isomap as Reducer
    elif model.upper() in {"MDS", "MULTIDIMENSIONALSCALING"}:
        from sklearn.manifold import MDS as Reducer  # noqa:N811
    elif model.upper() in {"SE", "SPECTRAL", "SPECTRALEMBEDDING"}:
        from sklearn.manifold import SpectralEmbedding as Reducer
    else:
        raise ValueError(f"invalid dimensionality reduction model: {model}")
    return Reducer, kwargs


def fit_model(graph: nx.Graph) -> Model:
    """Fit a node2vec model on the graph and wrap it."""
    # if you're on gensim 4.0.0 +, they renamed the size
    node2vec = Node2Vec(
        n_components=64,
        return_weight=2.3,  # from SEffNet
        neighbor_weight=1.9,  # from SEffNet
        walklen=8,  # from SEffNet
        epochs=8,  # from SEffNet
        w2vparams={
            "window": 4,  # from SEffNet
            "negative": 5,  # default
            "iter": 10,  # default
            # default from gensim,
            # see https://github.com/VHRanger/nodevectors/issues/34
            "batch_words": 10000,
        },
        verbose=True,
        keep_walks=False,
    )

    echo("fitting model")
    node2vec.fit(graph)

    return Model.from_node2vec(node2vec)


def get_undirected_graph_from_df(df: pd.DataFrame) -> nx.Graph:
    """Load a graph from an IndraNet TSV dataframe."""
    echo("Loading graph")
    graph = nx.from_pandas_edgelist(df)
    return process_graph(graph)


def process_graph(graph: nx.Graph) -> nx.Graph:
    """Process an undirected graph."""
    echo("Getting largest connected components")
    nodes = sorted(nx.connected_components(graph), key=len, reverse=True)[0]
    echo(f"Largest connected component has {len(nodes):,} " "nodes")
    echo("Inducing subgraph")
    giant = graph.subgraph(nodes)
    echo("Copying subgraph")
    giant = giant.copy()
    echo(
        f"Done inducing subgraph. Has {giant.number_of_nodes():,} nodes and {giant.number_of_edges():,} edges"
    )
    return giant
