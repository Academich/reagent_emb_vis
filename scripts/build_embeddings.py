import math
import argparse
from typing import Iterable
from functools import partial
from multiprocessing import Pool, cpu_count
from collections import Counter
from itertools import combinations

import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from umap import UMAP


def __smi_mol_counter(separator: str, smi_col: pd.Series):
    return smi_col.apply(lambda s: Counter(s.split(separator))).sum()


def get_reagent_statistics(smi_col: pd.Series, separator: str = ";", chunk_size: int = 1000) -> Counter:
    """
    Obtain frequency of the molecular occurrence among the reactants/reagents of all reactions
    in the form of a Counter. Uses process pool for faster processing.
    :param: chunk_size: size of a subset of the data to process at once
    :returns: One counter with reagent occurrences for all reactions in the specified pandas Series.
    """
    n_entries = smi_col.shape[0]
    f = partial(__smi_mol_counter, separator)

    with Pool(cpu_count()) as p:
        bigger_counters = p.map(f, [smi_col[i: i + chunk_size] for i in range(0, n_entries, chunk_size)])

    return np.sum(bigger_counters)


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

    data, rows, cols = [], [], []
    for (x, y), n in count_pair.items():
        pair_pmi_score = math.log((n / sum_pair) / (count_single[x] / sum_single) / (count_single[y] / sum_single))
        rows.append(reagent_to_index[x])
        cols.append(reagent_to_index[y])
        data.append(pair_pmi_score)
        rows.append(reagent_to_index[y])
        cols.append(reagent_to_index[x])
        data.append(pair_pmi_score)
    return csc_matrix((data, (rows, cols)))


def umap_projection(embeddings: np.array) -> pd.DataFrame:
    """
    Projects the given vectors to the plane
    and to the surface of a 3D sphere using UMAP.
    :param embeddings:
    :return: Pandas Dataframe with coordinates of projected points.
    """
    xy = UMAP(random_state=12345).fit_transform(embeddings)
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
    target = pd.read_csv(args.input, header=None)[0]
    print("Counting reagents...")
    reagent_occurrence_counter = get_reagent_statistics(target,
                                                        separator=args.separator).most_common(args.max_reagents)
    i2r = {i: smi for i, (smi, count) in enumerate(reagent_occurrence_counter) if count >= args.min_count}
    r2i = {v: k for k, v in i2r.items()}
    smiles = [None] * len(i2r)
    for i in i2r:
        smiles[i] = i2r[i]
    target = target.apply(lambda x: [r for r in x.split(args.separator) if r in r2i])

    print("Building PMI matrix...")
    pmi_scores = build_pmi_matrix(target, r2i)

    print("Factorizing PMI matrix...")
    embeddings, _, _ = svds(pmi_scores, k=args.emb_dim)
    norms = np.sqrt(np.sum(np.square(embeddings), axis=1, keepdims=True))
    embeddings /= np.maximum(norms, 1e-7)

    print("Building UMAP projection...")
    projection_result = umap_projection(embeddings)

    if args.standard is not None:
        standard_reagents = pd.read_csv(args.standard, index_col=[0], sep='\t')
        smiles_info = standard_smiles_information(pd.Series(smiles),
                                                  dict(standard_reagents.set_index("smiles")["name"]),
                                                  dict(standard_reagents.set_index("smiles")["class"]))
    else:
        smiles_info = pd.DataFrame(smiles)
        smiles_info.columns = ["smiles"]
    smiles_info["count"] = pd.Series([i for _, i in reagent_occurrence_counter])
    result = pd.concat((projection_result, smiles_info), axis=1)
    result.to_csv(args.output, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str,
                        help="Path to the file with reagent sets for reactions.")
    parser.add_argument("--min_count", type=int, default=0,
                        help="Minimum number of occurrences for a reagent. "
                             "Reagents that occur less than this number of times in the dataset are discarded.")
    parser.add_argument("--standard", type=str, default=None,
                        help="Path to the file with standard reagent information")
    parser.add_argument("--output", "-o", type=str,
                        help="A filepath to save the final report under.")
    parser.add_argument("--separator", type=str, default=";",
                        help="Separator between reagent SMILES in the input.")
    parser.add_argument("--emb_dim", "-d", type=int,
                        help="Embedding dimensionality for the SVD of the matrix of PMI scores between reagents.")
    parser.add_argument("--max_reagents", "-n", type=int, default=None,
                        help="Maximum number of the most common reagents to consider.")
    main(parser.parse_args())
