#!/usr/bin/env python3
"""
Molecular Analysis Script
This script demonstrates basic molecular data handling and visualization using ChEMBL data.
It connects molecular properties to concepts familiar to neuroscientists.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt
from chembl_webresource_client.new_client import new_client

def fetch_chembl_molecules(target_organism: str = "Homo sapiens", 
                         limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch molecules from ChEMBL database.
    
    Args:
        target_organism: Target organism (default: Homo sapiens)
        limit: Maximum number of molecules to fetch
        
    Returns:
        List of molecule dictionaries containing SMILES and properties
    """
    molecule = new_client.molecule
    activities = new_client.activity
    
    # Fetch molecules with activity data
    results = activities.filter(
        target_organism=target_organism,
        pchembl_value__isnull=False
    ).order_by('-pchembl_value')[:limit]
    
    molecules = []
    for result in results:
        mol = molecule.get(result['molecule_chembl_id'])
        molecules.append({
            'smiles': mol['molecule_structures']['canonical_smiles'],
            'pchembl': result['pchembl_value'],
            'target': result['target_pref_name']
        })
    
    return molecules

def calculate_molecular_properties(smiles: str) -> Dict[str, float]:
    """
    Calculate basic molecular properties using RDKit.
    
    Args:
        smiles: SMILES string of the molecule
        
    Returns:
        Dictionary of molecular properties
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    
    return {
        'molecular_weight': Descriptors.ExactMolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'aromatic_rings': Descriptors.NumAromaticRings(mol)
    }

def visualize_molecule(smiles: str, title: str = "") -> None:
    """
    Generate and display 2D visualization of a molecule.
    
    Args:
        smiles: SMILES string of the molecule
        title: Title for the visualization
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Could not parse SMILES: {smiles}")
        return
    
    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)
    
    # Create figure
    fig = plt.figure(figsize=(6, 6))
    img = Draw.MolToImage(mol)
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()

def main():
    """Main execution function."""
    print("Fetching molecules from ChEMBL...")
    molecules = fetch_chembl_molecules(limit=5)
    
    for mol in molecules:
        print("\n" + "="*50)
        print(f"Target: {mol['target']}")
        print(f"pChEMBL: {mol['pchembl']}")
        
        # Calculate and display properties
        props = calculate_molecular_properties(mol['smiles'])
        print("\nMolecular Properties:")
        for prop, value in props.items():
            print(f"{prop}: {value:.2f}")
        
        # Visualize molecule
        visualize_molecule(mol['smiles'], f"Target: {mol['target']}")

if __name__ == "__main__":
    main() 