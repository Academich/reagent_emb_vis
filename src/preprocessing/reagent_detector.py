import re
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import rdMolHash

from preprocessing.identifyReactants import extract_reactants
from preprocessing.fragment_grouping import group_fragments
from preprocessing.utils import canonicalize_reaction_remove_aam, canonicalize_reaction


def assign_reaction_roles_by_rdkit(smi: str) -> str:
    """
    Decides on reactant-reagent separation in a reaction SMILES.
    Uses the fingerprint-based technique implemented in RDKit.
    """
    left, center, right = smi.split(">")
    left = [i for i in left.split(".") if i]
    center = [i for i in center.split(".") if i]
    products = [i for i in right.split(".") if i]
    precursors = left + center
    left_new, center_new, right_new = extract_reactants(precursors, products)
    if not left_new:
        return smi
    return ".".join(left_new) + ">" + ".".join(center_new) + ">" + ".".join(right_new)


def assign_reaction_roles_by_aam(smi: str) -> str:
    """
    Molecules that appear in both sides of the reaction are reagents.
    Aside from that, all molecules that have atom map numbers that appear on the right side are reactants.
    :param smi:
    :return:
    """

    def match_found(mol: str, tgt_labels: list[str]) -> bool:
        aam_labels = pattern.findall(mol)
        for a in aam_labels:
            if a in tgt_labels:
                return True
        return False

    pattern = re.compile(":(\d+)\]")  # atom map numbers
    reactants, reagents = [], []
    left, center, right = smi.split(">")
    all_rs = [i for i in left.split(".") + center.split(".") if i]
    right_mols_set = set(right.split("."))
    for m in all_rs:
        if m in right_mols_set:
            reagents.append(m)
    all_rs = [m for m in all_rs if m not in reagents]

    tgt_aam_labels = pattern.findall(right)
    for m in all_rs:
        if match_found(m, tgt_aam_labels):
            reactants.append(m)
        else:
            reagents.append(m)

    return ">".join(
        (".".join(reactants),
         ".".join(reagents),
         right)
    )


def assign_reaction_roles_mixed(smi: str) -> str:
    """
    Decides on reactant-reagent separation in a reaction SMILES.
    Uses the fingerprint-based technique implemented in RDKit.
    Defaults to AAM-based role assignment on failure.
    """
    try:
        role_assigned_rxn = assign_reaction_roles_by_rdkit(smi)
    except AttributeError:
        return assign_reaction_roles_by_aam(smi)
    except RuntimeError:
        return assign_reaction_roles_by_aam(smi)
    except ValueError:
        return assign_reaction_roles_by_aam(smi)
    else:
        if role_assigned_rxn.startswith(">"):
            return assign_reaction_roles_by_aam(smi)
        return role_assigned_rxn


def extract_fragment_cxsmiles(cxsmiles: str) -> str:
    substring = cxsmiles[cxsmiles.index("f:"):]
    for i, char in enumerate(substring):
        if char in ("^", "c"):
            return substring[:i]
    return substring


def parse_frament_cxsmiles(fragment_cxsmiles: str) -> list[list[int]]:
    assert fragment_cxsmiles.startswith("f:")
    return [[int(i) for i in group.split(".")] for group in fragment_cxsmiles.split("f:")[1].split(",") if group]


def hash_mol(smi: str) -> str:
    return rdMolHash.MolHash(Chem.MolFromSmiles(smi), rdMolHash.HashFunction.CanonicalSmiles)


def update_reported_fragment_grouping(rsmi_orig: str, rsmi_reassigned, grouping: list[list[int]]) -> list[list[int]]:
    smi_sequence_orig: str = rsmi_orig.replace(">>", ".").replace(">", ".")
    smi_sequence_reassigned: str = rsmi_reassigned.replace(">>", ".").replace(">", ".")
    idx2hash_orig = {i: hash_mol(s) for i, s in enumerate(smi_sequence_orig.split("."))}
    hash2idx_reassigned = defaultdict(list)
    for i, s in enumerate(smi_sequence_reassigned.split(".")):
        hash2idx_reassigned[hash_mol(s)].append(i)

    new_grouping = []
    for orig_group in grouping:
        new_group = []
        for idx in orig_group:
            new_group.append(hash2idx_reassigned[idx2hash_orig[idx]].pop())
        new_group = sorted(new_group)
        new_grouping.append(new_group)
    return new_grouping


def adjust_reaction_roles_with_grouping(rsmi: str, grouping: list[list[int]]) -> str:
    smi_sequence: list[str] = rsmi.replace(">>", ".").replace(">", ".").split('.')

    left, center, right = rsmi.split(">")
    left = left.split('.')
    center = center.split('.')
    right = right.split('.')
    initial_sides = {}
    idx = 0
    for _ in left:
        initial_sides[idx] = "reactants"
        idx += 1
    for _ in center:
        initial_sides[idx] = "reagents"
        idx += 1
    for _ in right:
        initial_sides[idx] = "products"
        idx += 1

    final_sides = {"reactants": [], "reagents": [], "products": []}

    group2side = {}
    for j, group in enumerate(grouping):
        if all([initial_sides[i] == "reagents" for i in group]):
            group2side[j] = "reagents"
            continue
        if any([initial_sides[i] == "reactants" for i in group]):
            group2side[j] = "reactants"
            continue
        if any([initial_sides[i] == "products" for i in group]):
            group2side[j] = "products"

    processed_ids = set()
    for j, group in enumerate(grouping):
        final_sides[group2side[j]].append(".".join([smi_sequence[i] for i in group]))
        processed_ids = processed_ids | {i for i in group}
    for i in range(len(smi_sequence)):
        if i not in processed_ids:
            final_sides[initial_sides[i]].append(smi_sequence[i])

    return ".".join(final_sides["reactants"]) + ">" + ";".join(final_sides["reagents"]) + ">" + ".".join(
        final_sides["products"])


class EnhancedReactionRoleAssignment:
    def __init__(self,
                 role_assignment_mode: str,
                 canonicalization_mode: str | None = None,
                 grouping_mode: str | None = None):
        self.role_assignment_mode = role_assignment_mode
        self.canonicalization_mode = canonicalization_mode
        self.grouping_mode = grouping_mode

        assert grouping_mode in (None, "cxsmiles", "heuristic")
        assert canonicalization_mode in (None, "keep_aam", "remove_aam")
        assert role_assignment_mode in ("aam", "rdkit", "mixed")

    def assign_reaction_roles_basic(self, rsmi: str) -> str:
        if self.role_assignment_mode == "aam":
            return assign_reaction_roles_by_aam(rsmi)
        if self.role_assignment_mode == "rdkit":
            return assign_reaction_roles_by_rdkit(rsmi)
        if self.role_assignment_mode == "mixed":
            return assign_reaction_roles_mixed(rsmi)

    def assign_reaction_roles(self, rsmi: str) -> tuple[str, list[list[int]] | None]:
        if self.grouping_mode == "cxsmiles" and "|" in rsmi and "f:" in rsmi:
            rxn_and_cxsmiles = rsmi.split("|")
            rxn = rxn_and_cxsmiles[0].strip()
            cxsmiles = rxn_and_cxsmiles[1].strip()
            fragment_cxsmiles = extract_fragment_cxsmiles(cxsmiles)
            grouping = parse_frament_cxsmiles(fragment_cxsmiles)
            rxn_upd = self.assign_reaction_roles_basic(rxn)
            grouping_upd = update_reported_fragment_grouping(rxn, rxn_upd, grouping)
            return rxn_upd, grouping_upd
        else:
            return self.assign_reaction_roles_basic(rsmi.split("|")[0].strip()), None

    def group_fragments(self, rsmi: str, grouping: list[list[int]] | None) -> str:
        if grouping is None:
            if self.grouping_mode == "heuristic":
                left, center, right = rsmi.split(">")
                return ">".join((left, group_fragments(center), right))
            else:
                left, center, right = rsmi.split(">")
                return ">".join((left, ";".join([i for i in center.split(".")]), right))
        else:
            if self.grouping_mode == "cxsmiles":
                return adjust_reaction_roles_with_grouping(rsmi, grouping)
            if self.grouping_mode == "heuristic":
                left, center, right = rsmi.split(">")
                return ">".join((left, group_fragments(center), right))

    def canonicalize(self, rsmi) -> str:
        if self.canonicalization_mode is None:
            return rsmi
        if self.canonicalization_mode == "remove_aam":
            return canonicalize_reaction_remove_aam(rsmi)
        if self.canonicalization_mode == "keep_aam":
            return canonicalize_reaction(rsmi)

    def run(self, rsmi: str) -> str:
        rsmi_upd, grouping = self.assign_reaction_roles(rsmi)
        rsmi_grouped = self.group_fragments(rsmi_upd, grouping)
        return self.canonicalize(rsmi_grouped)


if __name__ == '__main__':
    s = "[OH-:1].[Na+].[Na].[O-]S([O-])(=O)=O.[Na+].[Na+].S([O-])([O-])(=O)=O.[Na+].[Na+].[C:18]([O-:21])([O-])=O.[K+].[K+].CCN=C=N[CH2:29][CH2:30][CH2:31]N(C)C.Cl.Cl.CN(C)CCCN=C=NCC.[CH:62]1[CH:67]=CC2N(O)N=N[C:64]=2[CH:63]=1.ON1[C:63]2[CH:64]=CC=[CH:67][C:62]=2N=N1.CC1C=CC(S(O)(=O)=O)=CC=1.C1(C)C=CC(S(O)(=O)=O)=CC=1.C([O-])(O)=O.[Na+].[Na].[H-].[H-].[H-].[H-].[Li+].[Al+3].[H-].[Al+3].[Li+].[H-].[H-].[H-]>CN(C=O)C.CN(C)C=O.CO.CO.C(Cl)(Cl)Cl.C(Cl)(Cl)Cl.CCOCC.C(OCC)C.CCOC(C)=O.C(OCC)(=O)C.CCO.C(O)C.C(Cl)Cl.ClCCl>[CH2:29]1[CH2:18][O:21][CH2:31][CH2:30]1.[O:1]1[CH2:64][CH2:63][CH2:62][CH2:67]1 |f:0.1.2,3.4.5.6.7.8,9.10.11,12.13.14.15,16.17,18.19,20.21.22,23.24.25.26.27.28.29.30.31.32.33.34,35.36,37.38,39.40,41.42,43.44,45.46,47.48,49.50|"
    role_master = EnhancedReactionRoleAssignment(
        role_assignment_mode="mixed",
        canonicalization_mode="remove_aam",
        grouping_mode="cxsmiles"
    )
    r = role_master.run(s)
    print(r)
