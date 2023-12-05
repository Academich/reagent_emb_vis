import math
from collections import Counter

from rdkit import Chem
from rdkit.Chem import MolFromSmiles

from preprocessing.utils import check_subset, counter_to_list, smi_charge

SOLVENTS = {
    "CC#N": "acetonitrile",
    "O": "water",
    "C[N+](=O)[O-]": "nitromethane",
    "Cc1ccccc1": "toluene",
    "c1ccccc1": "benzene",
    "Cc1ccccc1C": "o-xylene",
    "Cc1cccc(C)c1": "m-xylene",
    "Cc1ccc(C)cc1": "p-xylene",
    "c1ccncc1": "pyridine",
    "Clc1ccccc1": "chlorobenzene",
    "Clc1ccccc1Cl": "1,2-dichlorobenzene",
    "ClCCl": "dichloromethane",
    "CC(Cl)Cl": "1,1-dichloroethane",
    "ClCCCl": "1,2-dichloroethane",
    "ClC(Cl)Cl": "chloroform",
    "ClC(Cl)(Cl)Cl": "carbon tetrachloride",
    "C1CCOC1": "THF",
    "CC1CCCO1": "2-methyl-THF",
    "C1COCCO1": "1,4-dioxane",
    "CCCCC": "pentane",
    "CCCCCC": "hexane",
    "CCCCCCC": "heptane",
    "C1CCCCC1": "cyclohexane",
    "C1CCCC1": "cyclopentane",
    "CO": "methanol",
    "CCO": "ethanol",
    "CCCO": "1-propanol",
    "CCCCO": "n-butanol",
    "CC(C)O": "isopropanol",
    "CC(C)(C)O": "tert-butanol",
    "CC(C)CO": "isobutanol",
    "CC(C)=O": "acetone",
    "CCC(C)=O": "butanone",
    "CN(C)C=O": "DMF",
    "CN1CCCC1=O": "N-methyl-2-pyrrolidone",
    "CNC(C)=O": "N-methylacetamide",
    "CC(=O)N(C)C": "dimethylacetamide",
    "CS(C)=O": "DMSO",
    "CN(C)P(=O)(N(C)C)N(C)C": "HMPA",
    "OCCO": "ethylene glycol",
    "CC(=O)O": "acetic acid",
    "O=C(O)C(F)(F)F": "TFA",
    "CCOCC": "diethyl ether",
    "COCCOC": "1,2-DME",
    "COC(C)(C)C": "Methyl tert-butyl ether",
    "CCOC(C)=O": "ethyl acetate",
    "S=C=S": "carbon disulfide"
}

SPECIAL_METALS = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
                  "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
                  "La", "Ce", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Th", "U"}
SPECIAL_METALS = {MolFromSmiles(f'[{i}]').GetAtoms()[0].GetAtomicNum() for i in SPECIAL_METALS}

METALLOCENE_METALS = {"Ti", "Fe", "V", "Cr", "Mn", "Co", "Ni", "Y", "Zr", "Nb", "Mo"}
METALLOCENE_METALS = {MolFromSmiles(f'[{i}]').GetAtoms()[0].GetAtomicNum() for i in METALLOCENE_METALS}

COMMON_CATALYSTS = [
    Counter(["[Ti+3]", "[Cl-]", "[Cl-]", "[Cl-]"]),
    Counter(["Cl[Pd]Cl", "[Fe+2]", "c1ccc(P(c2ccccc2)[c-]2cccc2)cc1",
             "c1ccc(P(c2ccccc2)[c-]2cccc2)cc1"]),  # (DPPF)PdCl2
    Counter(["[Pd]", "c1ccc(P(c2ccccc2)c2ccccc2)cc1", "c1ccc(P(c2ccccc2)c2ccccc2)cc1", "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
             "c1ccc(P(c2ccccc2)c2ccccc2)cc1"]),  # Pd(PPh3)4
    # Pd2(dba)3
    Counter(["[Pd]", "[Pd]", "O=C(C=Cc1ccccc1)C=Cc1ccccc1", "O=C(C=Cc1ccccc1)C=Cc1ccccc1",
             "O=C(C=Cc1ccccc1)C=Cc1ccccc1"]),
    # Pd(dba)2
    Counter(["[Pd]", "O=C(C=Cc1ccccc1)C=Cc1ccccc1", "O=C(C=Cc1ccccc1)C=Cc1ccccc1"]),
    Counter(["[Ru]", "[Ru]", "[Ru]", "[C-]#[O+]", "[C-]#[O+]", "[C-]#[O+]", "[C-]#[O+]", "[C-]#[O+]", "[C-]#[O+]",
             "[C-]#[O+]", "[C-]#[O+]", "[C-]#[O+]", "[C-]#[O+]", "[C-]#[O+]", "[C-]#[O+]"]),  # Ru3(CO)12
    Counter(["[K+]", "[K+]", "[K+]", "[Fe+3]", "[C-]#N", "[C-]#N", "[C-]#N", "[C-]#N", "[C-]#N", "[C-]#N"]),
    # K3[Fe(CN)6]
    Counter(["Cl[Ru]Cl", "Cl[Ru]Cl", "c1ccccc1", "c1ccccc1"]),
    Counter(["CN(C)[P+](On1nnc2ccccc21)(N(C)C)N(C)C", "F[P-](F)(F)(F)(F)F"]),
    Counter(["[Na+]", "[K+]", "O=C([O-])[C@H](O)[C@@H](O)C(=O)[O-]"]),  # Potassium sodium L-(+)-tartrate
    Counter(["[Na+]", "[K+]", "O=C([O-])C(O)C(O)C(=O)[O-]"]),  # Potassium sodium tartrate
    Counter(["[Li+]", "[AlH4-]"]),
    Counter(["[Li+]", "[Al+3]", "[H-]", "[H-]", "[H-]", "[H-]"]),
    Counter(["CC(C)C[Al+]CC(C)C", "[H-]"]),
    Counter(["[Na+]", "[B+3]", "[H-]", "[H-]", "[H-]", "[H-]"]),
    Counter(["[NH4+]", "[Cl-]"]),
    Counter(["[Li+]", "[B+3]", "[H-]", "[H-]", "[H-]", "[H-]"]),
    Counter(["[Na+]", "[Na+]", "O=S([O-])([O-])=S"]),
    Counter(["[Pd+2]", "CC(=O)[O-]", "CC(=O)[O-]"]),
]


def contains_transition_metal(smi: str) -> bool:
    elements = {i.GetAtomicNum() for i in Chem.MolFromSmiles(smi).GetAtoms()}
    return len(elements & SPECIAL_METALS) > 0


def contains_metallocene_metal(smi: str) -> bool:
    elements = {i.GetAtomicNum() for i in Chem.MolFromSmiles(smi).GetAtoms()}
    return len(elements & METALLOCENE_METALS) > 0


# 1. Obtain a cation counter and an anion counter
# 2. Take the anion with the most total charge
# 3. Balance the anion with the appropriate cation
# 4. Decrease the cation counter and the anion counter
# 5. Repeat from 2 while both cation and anion counter are not empty

class FailedToBalanceChargesError(Exception):
    """
    A custom error for ChargeBalancer.
    Thrown when it cannot partition a SMILES string into salts.
    This is usually the case when a partition would require a salt
    with several different cations or anions.
    """
    pass


class ChargeBalancer:
    """
    A class that matches cations with appropriate anions in a SMILES string.
    Useful for explicit partition of haphazard anions and cations into salts.
    """

    def __init__(self, charged_smi: list[str]) -> None:
        self.cation_counter = Counter([s for s in charged_smi if smi_charge(s) > 0])
        self.anion_counter = Counter([s for s in charged_smi if smi_charge(s) < 0])

    def _balance(self, cation, anion):
        gcd = math.gcd(smi_charge(cation), -smi_charge(anion))
        n_cation = -smi_charge(anion) // gcd
        n_anion = smi_charge(cation) // gcd

        self._decrease_counters(cation, n_cation, anion, n_anion)

        return [cation] * n_cation + [anion] * n_anion

    def _decrease_counters(self, cation, n_cation, anion, n_anion):
        if n_cation > self.cation_counter[cation]:
            raise FailedToBalanceChargesError
        elif n_cation == self.cation_counter[cation]:
            self.cation_counter.pop(cation)
        else:
            self.cation_counter[cation] -= n_cation

        if n_anion > self.anion_counter[anion]:
            raise FailedToBalanceChargesError
        elif n_anion == self.anion_counter[anion]:
            self.anion_counter.pop(anion)
        else:
            self.anion_counter[anion] -= n_anion

    def _total_pos_charge(self, ca: str) -> int:
        return self.cation_counter[ca] * smi_charge(ca)

    def _total_neg_charge(self, an: str) -> int:
        return self.anion_counter[an] * smi_charge(an)

    def run(self) -> list[str]:
        result = []
        while self.cation_counter and self.anion_counter:
            # Balance the particle with the largest absolute charge
            anion = min(self.anion_counter, key=lambda x: (self._total_neg_charge(x), smi_charge(x)))
            cation = max(self.cation_counter, key=lambda x: (self._total_pos_charge(x), smi_charge(x)))
            anion_total_charge = -self._total_neg_charge(anion)
            cation_total_charge = self._total_pos_charge(cation)
            if min(anion_total_charge, cation_total_charge) > 1 and anion_total_charge == cation_total_charge:
                balanced = self._balance(cation, anion)
                result.append(".".join(balanced))
                continue

            if anion_total_charge >= cation_total_charge or len(self.cation_counter) == 1:
                chosen_cation = min(self.cation_counter,
                                    key=lambda x: -self._total_neg_charge(anion) - self._total_pos_charge(x))
                balanced = self._balance(chosen_cation, anion)
                result.append(".".join(balanced))

            else:
                chosen_anion = min(self.anion_counter,
                                   key=lambda x: -self._total_neg_charge(x) - self._total_pos_charge(cation))

                balanced = self._balance(cation, chosen_anion)
                result.append(".".join(balanced))

        return result


def group_fragments(smi: str, separator: str = ';') -> str | None:
    """
    Organizes reagents in a clean way.
    First, extracts common hardcoded catalysts which consist of several species
    Second, extracts appropriate cation-anion pairs
    Finally, treats all the remaining molecules as one reagent
    :param smi: SMILES of reagents separated by dots
    :param separator:
    :return: SMILES of reagents, in which all single reagent species are separated by semicolons
    """
    single_species = []
    solvents = []
    complex_catalysts = []
    charged_smi, uncharged_smi = [], []
    salt_smi = []

    smi = Chem.CanonSmiles(smi, useChiral=False)

    species_counter = Counter(smi.split("."))
    if "O" in species_counter:
        species_counter.pop("O")

    # Extracting common catalysts

    for catalyst in COMMON_CATALYSTS:
        if check_subset(species_counter, catalyst):
            complex_catalysts.append(".".join(counter_to_list(catalyst)))
        while check_subset(species_counter, catalyst):
            species_counter -= catalyst

    # Extracting solvents

    for solvent in SOLVENTS:
        if solvent in species_counter:
            solvents.append(solvent)
            species_counter.pop(solvent)

    contains_metallocene = any([contains_metallocene_metal(k) and smi_charge(k) != 0 for k in species_counter]) and any(
        [Chem.MolFromSmiles(k).HasSubstructMatch(Chem.MolFromSmiles("c1cc[cH-]c1")) for k in species_counter])
    if contains_metallocene:
        metallocene_like_particles = [i for i in smi.split('.') if (
                (contains_metallocene_metal(i) and smi_charge(i) != 0) or Chem.MolFromSmiles(i).HasSubstructMatch(
            Chem.MolFromSmiles("c1cc[cH-]c1")))]
        try:
            single_species += ChargeBalancer(metallocene_like_particles).run()
        except FailedToBalanceChargesError:
            return
        species_counter -= Counter(metallocene_like_particles)

    # Dealing with charged species

    # if the SMILES string has balanced charges
    if sum([smi_charge(i) for i in smi.split('.')]) == 0:
        for k, v in species_counter.items():
            if Chem.GetFormalCharge(Chem.MolFromSmiles(k)) == 0:
                uncharged_smi += [k] * v
            else:
                charged_smi += [k] * v
        if charged_smi:
            try:
                salt_smi = ChargeBalancer(charged_smi).run()
            except FailedToBalanceChargesError:
                return

            species_counter -= Counter(charged_smi)
    # if charges are unbalanced
    else:
        species_counter = Counter(
            {k: v for k, v in species_counter.items() if smi_charge(k) == 0 or contains_transition_metal(k)}
        )

    # Dealing with uncharged species

    # Removing all single species
    for k, v in species_counter.items():
        if v == 1:
            single_species.append(k)
            species_counter[k] = 0
    species_counter = Counter({k: v for k, v in species_counter.items() if v > 0})

    # If there is only one particle left, don't repeat it several times
    if len(species_counter) == 1:
        for k in species_counter:
            species_counter[k] = 1

    species_left = counter_to_list(species_counter)
    all_molecules = [".".join(species_left)] + complex_catalysts + single_species + salt_smi + solvents
    all_molecules = [Chem.CanonSmiles(m, useChiral=False) for m in all_molecules]
    return separator.join(all_molecules).strip(separator)
