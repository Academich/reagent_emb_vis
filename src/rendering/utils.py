import io
import base64

import pandas as pd
import numpy as np

from dash.html import Div

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdDepictor


def add_numerical_labels_for_classes(df: pd.DataFrame) -> pd.DataFrame:
    if "class" in df.columns:
        df["numerical_label"] = df["class"].map({v: i for i, v in enumerate(sorted(df["class"].unique()))})
    else:
        df["numerical_label"] = 0
        df["class"] = "All molecules"
        df["name"] = ''
    return df


def parse_uploaded_content(contents) -> pd.DataFrame | Div:
    """
    Parses a Pandas DataFrame that is uploaded to the app
    """
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        df = add_numerical_labels_for_classes(df)
    except Exception as e:
        return Div(["There was an error processing this file."])

    return df


def parse_contents(contents: str) -> pd.DataFrame:
    """
    Parses a Pandas DataFrame that is stored in the browser as JSON
    """
    data = eval(contents)
    return pd.DataFrame(data["data"], index=data["index"], columns=data["columns"])


def smarts_pattern(sma: str):
    return Chem.MolFromSmarts(sma)


def match_smiles_to_smarts(smi: str, smarts_pattern: str | None) -> bool:
    """
    Tells whether the given SMARTS pattern fits the given molecules.
    :param smi: Molecular SMILES
    :param smarts_pattern: SMARTS pattern
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    try:
        # Check if the SMILES matches the SMARTS pattern
        if smarts_pattern is not None:
            return mol.HasSubstructMatch(smarts_pattern)
    except Exception as e:
        print(f"Error: {e} on {mol}, {smarts_pattern}")
        return False


def smi2svg(smi: str) -> str:
    """
    Turns SMILES into SVG text ready to be embedded
    into the html code of the page.
    :param smi: Molecular SMILES
    """
    mol = Chem.MolFromSmiles(smi)
    rdDepictor.Compute2DCoords(mol)
    mc = Chem.Mol(mol.ToBinary())
    Chem.Kekulize(mc)
    drawer = Draw.MolDraw2DSVG(400, 400)
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')
    return svg


def to_cartesian_coordinates(theta: float, phi: float) -> tuple[float, float, float]:
    """
    Converts spherical coordinates on the unit sphere in 3D
    to cartesian coordinates.
    :param theta:
    :param phi:
    :return:
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def uniform_sphere_points(n_points: int = 100) -> tuple[float, float, float]:
    """
    Generates cartesian coordinates for uniformly distributed points
    on the unit sphere. The more points, the smoother the sphere surface.
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return x, y, z
