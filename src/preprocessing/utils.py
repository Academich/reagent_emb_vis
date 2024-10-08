import math
from typing import Callable, Iterable
from collections import Counter
from functools import lru_cache, partial
from itertools import combinations
from multiprocessing import Pool, cpu_count

from pandas import Series, concat
import numpy as np
from tqdm import tqdm

from scipy.sparse import csc_matrix

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

tqdm.pandas()


def canonical_remove_aam_mol(smi: str, stereo: bool = True) -> str:
    """
    Removes atom mapping from a Mol object using RDKit
    :param smi:
    :param stereo: A flag whether to keep stereochemical information in SMILES
    :return: Canonicalized SMILES with no atom mapping
    """
    mol = Chem.MolFromSmiles(smi)
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=stereo)


def canonicalize_reaction_remove_aam(smi: str) -> str:
    """
    Removes atom mapping from all molecules in a SMILES string using RDKit
    :param smi:
    :return: Canonicalized SMILES with no atom mapping
    """
    left, center, right = smi.split(">")
    left = [canonical_remove_aam_mol(i) for i in left.split('.')]
    center = [canonical_remove_aam_mol(i, stereo=False) for i in center.split(';')]
    right = [canonical_remove_aam_mol(i) for i in right.split('.')]
    return ".".join(left) + ">" + ";".join(center) + ">" + ".".join(right)


def canonicalize_reaction(smi: str) -> str:
    """
    Canonicalizes all molecules in a reaction, keeps atom mapping
    :param smi:
    :return:
    """
    left, center, right = smi.split(">")
    left = [Chem.CanonSmiles(i) for i in left.split('.')]
    center = [Chem.CanonSmiles(i, useChiral=False) for i in center.split(';')]
    right = [Chem.CanonSmiles(i) for i in right.split('.')]
    return ".".join(left) + ">" + ";".join(center) + ">" + ".".join(right)


# === Chemical curation

@lru_cache(maxsize=None)
def smi_charge(smi: str) -> int:
    """
    Gets the charge of a chemical particle written as SMILES
    :param smi: SMILES of one or several molecules or ions
    :return: formal charge of the string
    """
    return Chem.GetFormalCharge(Chem.MolFromSmiles(smi))


def check_subset(whole: Counter, part: Counter) -> bool:
    """
    Check if one counter is entirely contained within another counter
    :param whole: A bigger counter
    :param part: A smaller counter
    :return: True or False
    """
    return (whole & part) == part


def counter_to_list(smi_cnt: Counter) -> list[str]:
    """
    Turns a Counter with SMILES into a list with explicitly repeated SMILES separated by dots.
    Example: Counter({"[Fe+3]": 1, "[C-]#N": 3}) -> ["Fe+3", "[C-]#N.[C-]#N.[C-]#N"]
    :param smi_cnt:
    :return:
    """
    return [".".join([k] * v) for k, v in smi_cnt.items()]


# === Reagent statistics
def __smi_mol_counter(separator: str, smi_col: Series):
    return smi_col.apply(lambda s: Counter(s.split(separator))).sum()


def get_reagent_statistics(smi_col: Series, separator: str = ";", chunk_size: int = 1000) -> Counter:
    """
    Obtain frequency of the molecular occurrence among the reactants/reagents of all reactions
    in the form of a Counter. Uses process pool for faster processing.
    :param: chunk_size: size of a subset of the data to process at once
    :returns: One counter with reagent occurrences for all reactions in the specified pandas Series.
    """
    n_entries = smi_col.shape[0]
    if not n_entries:
        return Counter()
    f = partial(__smi_mol_counter, separator)

    with Pool(cpu_count()) as p:
        bigger_counters = p.map(f, [smi_col[i: i + chunk_size] for i in range(0, n_entries, chunk_size)])

    return np.sum(bigger_counters)


def build_pmi_dict(data: Iterable[list[str]]) -> dict[tuple[str, str], float]:
    """
    Builds a dictionary mapping a pair of (reagent) SMILES to its PMI score (point-wise mutual information).
    :param data: A collection of sets of reagents used together in one reaction.
    :return: A dictionary containing PMI scores.
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

    pair_pmi_scores = {}
    for (x, y), n in count_pair.items():
        pair_pmi_scores[(x, y)] = math.log2(
            (n / sum_pair) / (count_single[x] / sum_single) / (count_single[y] / sum_single))
    return pair_pmi_scores


def pmi_dict_to_sparse_matrix(smiles_pair_pmi_scores: dict[tuple[str, str], float],
                              reagent_to_index: dict[str, int]) -> csc_matrix:
    """
    Builds a sparse matrix with PMI scores (point-wise mutual information)
    between reagents.
    :param smiles_pair_pmi_scores: A dictionary mapping SMILES pairs to PMI scores.
    :param reagent_to_index: A dictionary mapping a reagent SMILES to its serial number.
    :return: A symmetric sparse matrix containing PMI scores.
    """
    data, rows, cols = [], [], []
    for (x, y), s in smiles_pair_pmi_scores.items():
        rows.append(reagent_to_index[x])
        cols.append(reagent_to_index[y])
        data.append(s)
        rows.append(reagent_to_index[y])
        cols.append(reagent_to_index[x])
        data.append(s)
    return csc_matrix((data, (rows, cols)))


# === Tools for faster data processing on CPU using pool of processes ===

def __parallelize(d: Series, func: Callable, num_of_processes: int) -> Series:
    data_split = np.array_split(d, num_of_processes)
    pool = Pool(num_of_processes)
    d = concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return d


def __run_on_subset(func: Callable, use_tqdm, data_subset):
    if use_tqdm:
        return data_subset.progress_apply(func)
    return data_subset.apply(func)


def parallelize_on_rows(d: Series, func, num_of_processes: int, use_tqdm=False) -> Series:
    """
    Makes the pd.Series.apply method parallel
    :param d: A Pandas Series of interest
    :param func: The function to apply on the Series
    :param num_of_processes: Number of processes to parallelize to
    :param use_tqdm: A flag whether to make the process verbose by using tqdm
    :return: The result of the .apply method as it would have been without parallelization
    """
    return __parallelize(d, partial(__run_on_subset, func, use_tqdm), num_of_processes)
