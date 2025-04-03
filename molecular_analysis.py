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
import os

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

def assess_cns_penetration(props: Dict[str, float]) -> Dict[str, Any]:
    """
    Assess potential for CNS penetration using simple rules.
    
    Args:
        props: Dictionary of molecular properties
        
    Returns:
        Dictionary containing assessment results and explanations
    """
    assessment = {
        'bbb_profile': '',
        'explanations': []
    }
    
    # Lipinski's Rule of 5 for CNS drugs
    if props['molecular_weight'] <= 400:
        assessment['explanations'].append("Molecular weight ≤ 400 Da (favorable)")
    else:
        assessment['explanations'].append("Molecular weight > 400 Da (less favorable)")
    
    if props['logp'] <= 3:
        assessment['explanations'].append("LogP ≤ 3 (favorable)")
    else:
        assessment['explanations'].append("LogP > 3 (less favorable)")
    
    if props['hbd'] <= 3:
        assessment['explanations'].append("HBD ≤ 3 (favorable)")
    else:
        assessment['explanations'].append("HBD > 3 (less favorable)")
    
    if props['rotatable_bonds'] <= 8:
        assessment['explanations'].append("Rotatable bonds ≤ 8 (favorable)")
    else:
        assessment['explanations'].append("Rotatable bonds > 8 (less favorable)")
    
    # Overall assessment
    favorable_count = sum(1 for exp in assessment['explanations'] if "favorable" in exp)
    if favorable_count >= 3:
        assessment['bbb_profile'] = "Favorable BBB profile"
    else:
        assessment['bbb_profile'] = "Less favorable BBB profile"
    
    return assessment

def visualize_molecule(smiles: str, title: str = "", save_dir: str = "molecule_plots") -> None:
    """
    Generate and display 2D visualization of a molecule.
    
    Args:
        smiles: SMILES string of the molecule
        title: Title for the visualization
        save_dir: Directory to save the visualization
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
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    plt.savefig(os.path.join(save_dir, f"{safe_title}.png"))
    plt.close()

def plot_property_comparison(molecules: List[Dict[str, Any]], properties: List[str]) -> None:
    """
    Create bar charts comparing properties across molecules.
    
    Args:
        molecules: List of molecule dictionaries
        properties: List of property names to compare
    """
    # Prepare data for plotting
    data = {prop: [] for prop in properties}
    labels = []
    
    for mol in molecules:
        props = calculate_molecular_properties(mol['smiles'])
        labels.append(mol['target'][:20] + '...')  # Truncate long labels
        for prop in properties:
            data[prop].append(props.get(prop, 0))
    
    # Create subplots for each property
    n_props = len(properties)
    fig, axes = plt.subplots(1, n_props, figsize=(5*n_props, 5))
    if n_props == 1:
        axes = [axes]
    
    for ax, prop in zip(axes, properties):
        ax.bar(range(len(data[prop])), data[prop])
        ax.set_title(prop)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('molecule_plots/property_comparison.png')
    plt.close()

def main():
    """Main execution function."""
    print("Fetching molecules from ChEMBL...")
    molecules = fetch_chembl_molecules(limit=5)
    
    # Create directory for plots
    os.makedirs('molecule_plots', exist_ok=True)
    
    # Properties to compare across molecules
    properties_to_compare = ['molecular_weight', 'logp', 'hbd', 'hba']
    
    for mol in molecules:
        print("\n" + "="*50)
        print(f"Target: {mol['target']}")
        print(f"pChEMBL: {mol['pchembl']}")
        
        # Calculate and display properties
        props = calculate_molecular_properties(mol['smiles'])
        print("\nMolecular Properties:")
        for prop, value in props.items():
            print(f"{prop}: {value:.2f}")
        
        # Assess CNS penetration
        cns_assessment = assess_cns_penetration(props)
        print(f"\nCNS Penetration Assessment: {cns_assessment['bbb_profile']}")
        print("Details:")
        for explanation in cns_assessment['explanations']:
            print(f"- {explanation}")
        
        # Visualize molecule
        visualize_molecule(mol['smiles'], f"Target: {mol['target']}")
    
    # Create property comparison plots
    print("\nGenerating property comparison plots...")
    plot_property_comparison(molecules, properties_to_compare)
    print("Plots saved in 'molecule_plots' directory")

if __name__ == "__main__":
    main() 