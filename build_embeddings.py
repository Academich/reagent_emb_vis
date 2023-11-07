import math
import argparse
from typing import Iterable
import json
from collections import Counter
from itertools import combinations
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from umap import UMAP


def build_pmi_matrix(data: Iterable[list[str]],
                     reagent_to_index: dict[str, int]) -> csc_matrix:
    """
    Builds a sparse matrix of PMI scores (point-wise mutual information)
    between reagents given the collection of co-occuring sets of reagents.
    :param data: A collection of sets reagents used together in one reaction.
    :param reagent_to_index: A dictionary mapping a reagent SMILES to its serial number.
    :return: A sparse matrix containing PMI scores.
    """
    count_single = Counter()
    count_pair = Counter()
    for entry in data:
        for reagent in entry:
            count_single[reagent] += 1
        for reagent_1, reagent_2 in map(sorted, combinations(entry, 2)):
            count_pair[(reagent_1, reagent_2)] += 1

    sum_single = sum(count_single.values())
    sum_pair = sum(count_pair.values())

    pmi_samples = {}
    data, rows, cols = [], [], []
    for (x, y), n in count_pair.items():
        rows.append(reagent_to_index[x])
        cols.append(reagent_to_index[y])
        data.append(math.log((n / sum_pair) / (count_single[x] / sum_single) / (count_single[y] / sum_single)))
        pmi_samples[(x, y)] = data[-1]
    return csc_matrix((data, (rows, cols)))


def umap_projection(embeddings: np.array) -> pd.DataFrame:
    """
    Projects the given vectors to the plane
    and to the surface of a 3D sphere using UMAP.
    :param embeddings:
    :return: Pandas Dataframe with coordinates of projected points.
    """
    xy = UMAP().fit_transform(embeddings)
    sphere_mapper = UMAP(output_metric='haversine', random_state=12345, n_neighbors=30).fit(embeddings)
    result = pd.DataFrame(np.hstack((xy, sphere_mapper.embedding_)))
    result.columns = ["x", "y", "theta", "phi"]
    return result


def standard_smiles_information(smiles: pd.Series,
                                smiles_to_names: dict[str, str],
                                smiles_to_roles: dict[str, str]) -> pd.DataFrame:
    """
    Adds information about reagents.
    :param smiles: Series with SMILES.
    :param smiles_to_names: Dictionary mapping SMILES to molecule names.
    :param smiles_to_roles: Dictionary mapping reagent SMILES to their
    standard reaction roles, e.g. catalysts, solvents, etc.
    :return: DataFrame with SMILES, names and reaction roles.
    """
    roles = smiles.map(smiles_to_roles, na_action='ignore').fillna("unk")
    names = smiles.map(smiles_to_names, na_action='ignore').fillna("???")
    result = pd.concat((smiles, roles, names), axis=1)
    result.columns = ["smiles", "class", "name"]
    return result


def main(args) -> None:
    target = pd.read_csv(args.input, header=None)[0].str.split(args.separator)
    with open(args.reagents) as f:
        i2r = {int(k): v for k, v in json.load(f).items()}
        r2i = {v: k for k, v in i2r.items()}
    smiles = [None] * len(i2r)
    for i in i2r:
        smiles[i] = i2r[i]
    standard_reagents = pd.read_csv(args.standard, index_col=[0], sep='\t')

    print("Building PMI matrix...")
    pmi_scores = build_pmi_matrix(target, r2i)

    print("Factorizing PMI matrix...")
    embeddings, _, _ = svds(pmi_scores, k=args.emb_dim)
    norms = np.sqrt(np.sum(np.square(embeddings), axis=1, keepdims=True))
    embeddings /= np.maximum(norms, 1e-7)

    print("Building UMAP projection...")
    projection_result = umap_projection(embeddings)

    smiles_info = standard_smiles_information(pd.Series(smiles),
                                              dict(standard_reagents.set_index("smiles")["name"]),
                                              dict(standard_reagents.set_index("smiles")["class"]))
    result = pd.concat((projection_result, smiles_info), axis=1)
    result.to_csv(args.output, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str,
                        help="Path to the file with reagent sets for reactions.")
    parser.add_argument("--reagents", "-r", type=str,
                        help="Path to the JSON file with unique reagents that occur in the input.")
    parser.add_argument("--standard", type=str, default=None,
                        help="Path to the file with standard reagent information")
    parser.add_argument("--output", "-o", type=str,
                        help="A filepath to save the final report under.")
    parser.add_argument("--separator", type=str, default=";",
                        help="Separator between reagent SMILES in the input.")
    parser.add_argument("--emb_dim", "-d", type=int,
                        help="Embedding dimensionality for the SVD of the matrix of PMI scores between reagents.")
    main(parser.parse_args())
