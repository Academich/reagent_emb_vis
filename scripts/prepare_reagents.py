from pathlib import Path
from argparse import ArgumentParser, Namespace

from multiprocessing import cpu_count

import pandas as pd

from rdkit import RDLogger

import preprocessing.utils as ut
from preprocessing.reagent_detector import EnhancedReactionRoleAssignment
from preprocessing.reagent_replacement import REAGENT_STANDARD_REPLACEMENT


def is_long_reaction(rxn: str, separator: str = ";") -> bool:
    """
    Checks if a reaction is too long.
    A reaction is too long if it has more than 10 reactants and reagents or more than 5 reagents
    :param rxn: Reaction SMILES
    :param separator: Separator between reagents
    :return: True if there are more than 10 precursors in the reaction, False otherwise
    """
    left, center, _ = rxn.split(">")
    left_len = len(left.split("."))
    center_len = len(center.split(separator))
    return center_len > 5 or left_len + center_len > 10


def is_trivial_reaction(rxn: str) -> bool:
    """
    Checks if there is an overlap between reaction products and its precursors.
    Works best with reaction SMILES that have only one reported product.
    :param rxn: Reaction SMILES
    :return: True if some product appears among reactants or reagents, False otherwise
    """
    left, center, right = rxn.split(">")
    reactant_set = set(left.split("."))
    reagent_set = set(center.split("."))
    product_set = set(right.split("."))
    return bool(reactant_set & product_set) or bool(reagent_set & product_set)


def report_reagent_statistics(reagent_occurrences: dict[str, int], minimum_repeats: int) -> None:
    """
    Print some statistics regarding reagents in the dataset.
    :param reagent_occurrences: A dictionary mapping a reagent SMILES string
    to the number of times that string occurs in the dataset.
    :param minimum_repeats: Minimal required number of times for a reagent to occur to not be deleted.
    """
    frequent_reagents = {k: v for k, v in reagent_occurrences.items() if v >= minimum_repeats}
    reagents_encountered_once = {k for k, v in reagent_occurrences.items() if v == 1}
    print("Unique reagents: %d" % len(reagent_occurrences))
    print("Reagents occurring more than %d times: %d (%.3f%%)" % (minimum_repeats,
                                                                  len(frequent_reagents),
                                                                  len(frequent_reagents) * 100 / len(
                                                                      reagent_occurrences)))
    print("Reagents encountered only once: %d (%.3f%%)" % (len(reagents_encountered_once),
                                                           len(reagents_encountered_once) * 100 / len(
                                                               reagent_occurrences)))


def standardize_reagents(smi: str,
                         replacement_dict: dict[str, list[str]],
                         separator: str) -> str:
    updated_smiles = []
    for m in smi.split(separator):
        if m in replacement_dict:
            updated_smiles = updated_smiles + replacement_dict[m]
        else:
            updated_smiles.append(m)
    return separator.join(updated_smiles)


def remove_certain_reagents(smi: str,
                            species_to_remove: set[str],
                            separator: str) -> str:
    return separator.join([m for m in smi.split(separator) if m not in species_to_remove])


def remove_bound_water(smi: str,
                       separator: str) -> str:
    return separator.join(
        [".".join([i for i in m.split(".") if i != "O"]) if m != "O" else m for m in smi.split(separator)])


def main(args: Namespace) -> None:
    """
    Makes input files for the main script with a model and a data module.
    Takes raw reactions, like USPTO SMILES, as input.
    :param args: Command line arguments. For more information, run 'python3 prepare_data.py --help'
    """
    input_data = Path(args.input_data)
    print("Reading reactions from %s" % input_data)
    reactions: pd.Series = pd.read_csv(input_data,
                                       sep=args.csv_separator,
                                       skiprows=args.skiprows,
                                       usecols=[args.source_column])[args.source_column]
    print("Number of reactions: %d" % reactions.shape[0])

    role_master = EnhancedReactionRoleAssignment(
        role_assignment_mode=args.reagents,
        canonicalization_mode=args.canonicalization,
        grouping_mode=args.fragment_grouping
    )

    standardized_reactions = ut.parallelize_on_rows(reactions, role_master.run, args.n_jobs, use_tqdm=args.verbose)

    # Dropping bad reactions

    n_with_dupl = standardized_reactions.shape[0]
    standardized_reactions.drop_duplicates(inplace=True)
    n_no_dupl = standardized_reactions.shape[0]
    print(f"Dropping duplicates: {n_with_dupl - n_no_dupl}. Reactions left: {n_no_dupl}")

    reactions_without_reagents = standardized_reactions.apply(lambda x: ">>" in x)
    print(f"Removing reactions without reagents: {reactions_without_reagents.sum()}")
    standardized_reactions = standardized_reactions[~reactions_without_reagents]

    reactions_too_long = standardized_reactions.apply(is_long_reaction)
    print(f"Removing reactions that are too long: {reactions_too_long.sum()}")
    standardized_reactions = standardized_reactions[~reactions_too_long]

    reactions_trivial = standardized_reactions.apply(is_trivial_reaction)
    print(f"Removing reactions where product appears among reactants or reagents: {reactions_trivial.sum()}")
    standardized_reactions = standardized_reactions[~reactions_trivial]
    if standardized_reactions.empty:
        return print("All the reactions were removed :(")

    # Processing reagents
    reaction_parts = standardized_reactions.str.split(">", expand=True)
    reactants = reaction_parts[0]
    reagents = reaction_parts[1]
    products = reaction_parts[2]

    # Delete entries with empty reactants or products
    invalid_reactions = (reactants == "") | (products == "")
    print(f"Removing reactions with no reactants or no products: {invalid_reactions.sum()}")
    reactants = reactants[~invalid_reactions]
    reagents = reagents[~invalid_reactions]
    products = products[~invalid_reactions]

    print("Removing bound water")
    reagents = reagents.apply(lambda x: remove_bound_water(x, separator=";"))

    print("Standardizing reagents")
    reagents = reagents.apply(
        lambda x: standardize_reagents(x, replacement_dict=REAGENT_STANDARD_REPLACEMENT, separator=";"))

    # Drop duplicate molecules from reagents
    reagents = reagents.apply(lambda x: ";".join(set(x.split(';'))))

    # Drop molecules that rarely occur in reagents we ended up with after organizing them
    reagent_occurrence_counter = ut.get_reagent_statistics(reagents, separator=";")
    print("In organized reagents:")
    report_reagent_statistics(reagent_occurrence_counter, args.min_reagent_occurrences)
    print("Removing rare reagents")
    rare_reagents = {k for k, v in reagent_occurrence_counter.items() if v < args.min_reagent_occurrences}

    print("Removing uncharged reagent species")
    charged_reagent_species = {k for k in reagent_occurrence_counter if ut.smi_charge(k) != 0}

    undesirable_reagents = rare_reagents | charged_reagent_species
    reagents = reagents.apply(
        lambda smi: remove_certain_reagents(smi, undesirable_reagents, separator=";")
    )

    # Delete entries with empty reagents
    reagents_empty = reagents == ""
    print(f"Removing reactions with no reagents: {reagents_empty.sum()}")
    reactants = reactants[~reagents_empty]
    reagents = reagents[~reagents_empty]
    products = products[~reagents_empty]

    # Save files
    print("Saving files")
    save_path = Path(args.output_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    reactions_final = reactants + ">>" + products
    save_path_reactions = (save_path / f"transformations-{reactions_final.shape[0]}").with_suffix(".txt")
    reactions_final.to_csv(
        save_path_reactions,
        index=False,
        header=False,
    )
    save_path_reagents = (save_path / f"reagents-{reagents.shape[0]}").with_suffix(".txt")
    reagents.to_csv(
        save_path_reagents,
        index=False,
        header=False,
    )
    print("Transformations saved in %s" % save_path_reactions)
    print("Reagents saved in %s" % save_path_reagents)


if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')

    parser = ArgumentParser()
    parser.add_argument("--input_data", "-i", type=str, required=True,
                        help="Path to the raw data (.csv) that needs preprocessing.")
    parser.add_argument("--output_dir", "-o", type=str, default="", help="Name of the directory with tokenized files.")
    parser.add_argument("--csv_separator", type=str, default="\t",
                        help="Separator in the input .csv file.")
    parser.add_argument("--skiprows", type=int, default=0,
                        help="How many rows to skip from the beginning of the file. "
                             "Useful in case the file starts with comments.")
    parser.add_argument("--source_column", "-c", type=str,
                        help="Name of the column in the input that needs preprocessing.",
                        default="ReactionSmiles")
    parser.add_argument("--reagents", type=str, required=True, choices=["aam", "rdkit", "mixed"],
                        help="The way of deciding which molecules in a reaction are reagents. "
                             "Possible choices: aam, rdkit. The first relies on atom mapping, "
                             "the second on a fingerprint-based technique from Schneider et al. 2016")
    parser.add_argument("--fragment_grouping", type=str, default=None, choices=["cxsmiles", "heuristic"],
                        help="How to do fragment grouping in reactions. Options: cxsmiles, heuristic. "
                             "If not provided, no reagent grouping will be used")
    parser.add_argument("--canonicalization", type=str, default=None, choices=["keep_aam", "remove_aam"],
                        help="Specifies the way of reaction canonicalization - with atom mapping or without it."
                             "Atom mapping might be necessary for some reaction encoders. "
                             "It is recommended to use 'remove_aam' together when --reagents=rdkit. "
                             "If not specified, reactions are not canonicalized.")
    parser.add_argument("--n_jobs", type=int, default=cpu_count(),
                        help="Number of processes to use in parallelized functions.")
    parser.add_argument("--verbose", action="store_true",
                        help="Whether to show progress bar of reagent preprocessing.")
    parser.add_argument("--min_reagent_occurrences", type=int, default=None,
                        help="If not None, all reagent with number of occurrences less than this "
                             "number will be removed.")

    main(parser.parse_args())
