#!/usr/bin/env python3
"""
Molecular Analysis Script
This script demonstrates molecular data handling, visualization, and analysis using ChEMBL data.
It connects molecular properties to concepts familiar to neuroscientists, with a focus on CNS drug discovery.
"""

from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, GraphDescriptors
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt
from chembl_webresource_client.new_client import new_client
import os
import json
from pathlib import Path

# Constants
DEFAULT_SAVE_DIR = "molecule_plots"
CNS_DRUG_LIKENESS_THRESHOLDS = {
    'MW': (150, 450),
    'logP': (1, 3),
    'HBD': (0, 3),
    'TPSA': (0, 90),
    'RotatableBonds': (0, 10)
}

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

def visualize_molecule(smiles: str, title: str = "", save_dir: str = DEFAULT_SAVE_DIR) -> None:
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

def plot_property_comparison(molecules: List[Dict[str, Any]], properties: List[str], save_dir: str = DEFAULT_SAVE_DIR) -> None:
    """
    Create bar charts comparing properties across molecules.
    
    Args:
        molecules: List of molecule dictionaries
        properties: List of property names to compare
        save_dir: Directory to save the plots
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
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'property_comparison.png'))
    plt.close()

def analyze_molecular_descriptors(smiles: str) -> Dict[str, Any]:
    """
    Analyze molecular descriptors with CNS drug discovery focus.
    
    Args:
        smiles: SMILES string of the molecule
        
    Returns:
        Dictionary containing various molecular descriptors
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # Physicochemical properties (CNS-relevant)
    descriptors = {
        # Lipophilicity (related to blood-brain barrier penetration)
        'logP': Descriptors.MolLogP(mol),
        
        # Molecular weight (affects BBB penetration)
        'MW': Descriptors.ExactMolWt(mol),
        
        # Hydrogen bond donors (affects BBB penetration)
        'HBD': Descriptors.NumHDonors(mol),
        
        # Hydrogen bond acceptors (affects BBB penetration)
        'HBA': Descriptors.NumHAcceptors(mol),
        
        # Topological polar surface area (BBB penetration)
        'TPSA': Descriptors.TPSA(mol),
        
        # Rotatable bonds (flexibility, related to target binding)
        'RotatableBonds': Descriptors.NumRotatableBonds(mol),
        
        # Aromatic rings (pharmacophore features)
        'AromaticRings': Descriptors.NumAromaticRings(mol),
    }
    
    # Generate ECFP4 fingerprint (2048 bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    
    # Calculate fingerprint statistics
    descriptors['fingerprint_density'] = sum(fp) / len(fp)
    
    # Add topological indices
    descriptors.update(calculate_topological_indices(mol))
    
    return descriptors

def assess_cns_drug_likeness(descriptors: Dict[str, float]) -> Dict[str, Any]:
    """
    Assess CNS drug-likeness based on molecular descriptors.
    
    Args:
        descriptors: Dictionary of molecular descriptors
        
    Returns:
        Dictionary containing CNS drug-likeness assessment
    """
    # CNS drug-likeness criteria (based on literature)
    criteria = {}
    for prop, (min_val, max_val) in CNS_DRUG_LIKENESS_THRESHOLDS.items():
        if prop in descriptors:
            criteria[f'{prop}_optimal'] = min_val <= descriptors[prop] <= max_val
    
    # Calculate overall CNS drug-likeness score
    score = sum(criteria.values()) / len(criteria) if criteria else 0
    
    return {
        'cns_drug_likeness_score': score,
        'criteria_met': criteria,
        'interpretation': 'High CNS penetration potential' if score >= 0.8 else 
                         'Moderate CNS penetration potential' if score >= 0.6 else 
                         'Low CNS penetration potential'
    }

def calculate_wiener_index(mol: Chem.Mol) -> float:
    """
    Calculate the Wiener index for a molecule.
    The Wiener index is the sum of all shortest paths between atoms.
    Analogous to average path length in neural networks.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Wiener index value
    """
    # Get the number of atoms
    n_atoms = mol.GetNumAtoms()
    
    # Calculate the distance matrix (shortest paths between all atoms)
    distance_matrix = Chem.GetDistanceMatrix(mol)
    
    # Sum all distances to get the Wiener index
    wiener_index = np.sum(distance_matrix)
    
    return float(wiener_index)

def calculate_balaban_index(mol: Chem.Mol) -> float:
    """
    Calculate the Balaban connectivity index (J).
    Similar to clustering coefficient in neural networks.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Balaban index value
    """
    # Get the number of atoms and bonds
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()
    
    # Calculate edge connectivity
    edge_connectivity = 0
    for bond in mol.GetBonds():
        # Get the atoms connected by this bond
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        
        # Calculate the sum of the reciprocals of the square roots of the 
        # valences of the atoms connected by the bond
        edge_connectivity += 1 / np.sqrt(atom1.GetDegree() * atom2.GetDegree())
    
    # Calculate the Balaban index
    if n_bonds > 0:
        balaban_index = (n_bonds / (n_bonds - n_atoms + 2)) * edge_connectivity
    else:
        balaban_index = 0.0
    
    return float(balaban_index)

def calculate_zagreb_indices(mol: Chem.Mol) -> Tuple[float, float]:
    """
    Calculate the Zagreb indices (M1 and M2).
    Related to node degree distribution in networks.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Tuple of (M1, M2) Zagreb indices
    """
    # M1: sum of squared vertex degrees
    m1 = 0
    # M2: sum of products of degrees of adjacent vertices
    m2 = 0
    
    # Calculate M1
    for atom in mol.GetAtoms():
        degree = atom.GetDegree()
        m1 += degree * degree
    
    # Calculate M2
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        m2 += atom1.GetDegree() * atom2.GetDegree()
    
    return float(m1), float(m2)

def calculate_connectivity_indices(mol: Chem.Mol) -> Tuple[float, float, float]:
    """
    Calculate the connectivity indices (Chi0v, Chi1v, Chi2v).
    Similar to edge connectivity in neural networks.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Tuple of (Chi0v, Chi1v, Chi2v) connectivity indices
    """
    # Chi0v: 0th order valence connectivity
    chi0v = 0
    # Chi1v: 1st order valence connectivity
    chi1v = 0
    # Chi2v: 2nd order valence connectivity
    chi2v = 0
    
    # Calculate Chi0v
    for atom in mol.GetAtoms():
        chi0v += 1 / np.sqrt(atom.GetDegree())
    
    # Calculate Chi1v
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        chi1v += 1 / np.sqrt(atom1.GetDegree() * atom2.GetDegree())
    
    # Calculate Chi2v (simplified)
    for atom in mol.GetAtoms():
        if atom.GetDegree() >= 2:
            neighbors = [neighbor for neighbor in atom.GetNeighbors()]
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    chi2v += 1 / np.sqrt(neighbors[i].GetDegree() * neighbors[j].GetDegree())
    
    return float(chi0v), float(chi1v), float(chi2v)

def calculate_kappa_indices(mol: Chem.Mol) -> Tuple[float, float, float]:
    """
    Calculate the Kappa shape indices (kappa1, kappa2, kappa3).
    Capture molecular shape and branching.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Tuple of (kappa1, kappa2, kappa3) Kappa indices
    """
    # Get the number of atoms
    n_atoms = mol.GetNumAtoms()
    
    # Calculate the number of paths of length 1, 2, and 3
    p1 = mol.GetNumBonds()
    p2 = 0
    p3 = 0
    
    # Calculate p2 (paths of length 2)
    for atom in mol.GetAtoms():
        if atom.GetDegree() >= 2:
            p2 += atom.GetDegree() * (atom.GetDegree() - 1) // 2
    
    # Calculate p3 (paths of length 3)
    for atom in mol.GetAtoms():
        if atom.GetDegree() >= 2:
            neighbors = [neighbor for neighbor in atom.GetNeighbors()]
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if mol.GetBondBetweenAtoms(neighbors[i].GetIdx(), neighbors[j].GetIdx()) is None:
                        p3 += 1
    
    # Calculate Kappa indices
    if n_atoms > 1:
        kappa1 = (n_atoms - 1) * (n_atoms - 1) / p1 if p1 > 0 else 0
        kappa2 = (n_atoms - 1) * (n_atoms - 2) * (n_atoms - 2) / p2 if p2 > 0 else 0
        kappa3 = (n_atoms - 1) * (n_atoms - 3) * (n_atoms - 3) / p3 if p3 > 0 else 0
    else:
        kappa1 = kappa2 = kappa3 = 0.0
    
    return float(kappa1), float(kappa2), float(kappa3)

def calculate_randic_index(mol: Chem.Mol) -> float:
    """
    Calculate the Randic connectivity index.
    Measures branching patterns.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Randic index value
    """
    randic_index = 0
    
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        randic_index += 1 / np.sqrt(atom1.GetDegree() * atom2.GetDegree())
    
    return float(randic_index)

def calculate_bertz_complexity(mol: Chem.Mol) -> float:
    """
    Calculate the Bertz complexity index.
    Measures molecular complexity.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Bertz complexity index value
    """
    # This is a simplified version of the Bertz complexity index
    # The actual implementation is more complex
    
    # Count the number of atoms, bonds, and rings
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()
    n_rings = Chem.GetSSSR(mol).Count()
    
    # Calculate a simple complexity score
    complexity = n_atoms + n_bonds + 2 * n_rings
    
    return float(complexity)

def calculate_topological_indices(mol: Chem.Mol) -> Dict[str, float]:
    """
    Calculate key topological indices for a molecule.
    These indices capture structural connectivity patterns similar to neural network metrics.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary containing various topological indices
    """
    # Calculate basic topological indices
    indices = {
        # Wiener Index (sum of all shortest paths between atoms)
        # Analogous to average path length in neural networks
        'wiener_index': calculate_wiener_index(mol),
        
        # Balaban Index (connectivity index)
        # Similar to clustering coefficient in neural networks
        'balaban_index': calculate_balaban_index(mol),
    }
    
    # Zagreb Indices (sum of squared vertex degrees)
    # Related to node degree distribution in networks
    zagreb_m1, zagreb_m2 = calculate_zagreb_indices(mol)
    indices['zagreb_m1'] = zagreb_m1
    indices['zagreb_m2'] = zagreb_m2
    
    # Connectivity Indices
    # Similar to edge connectivity in neural networks
    chi0v, chi1v, chi2v = calculate_connectivity_indices(mol)
    indices['chi0v'] = chi0v  # 0th order valence connectivity
    indices['chi1v'] = chi1v  # 1st order valence connectivity
    indices['chi2v'] = chi2v  # 2nd order valence connectivity
    
    # Kappa Shape Indices
    # Capture molecular shape and branching
    kappa1, kappa2, kappa3 = calculate_kappa_indices(mol)
    indices['kappa1'] = kappa1
    indices['kappa2'] = kappa2
    indices['kappa3'] = kappa3
    
    # Randic Index (connectivity index)
    # Measures branching patterns
    indices['randic_index'] = calculate_randic_index(mol)
    
    # Bertz Complexity Index
    # Measures molecular complexity
    indices['bertz_index'] = calculate_bertz_complexity(mol)
    
    return indices

def analyze_topological_patterns(indices: Dict[str, float]) -> Dict[str, str]:
    """
    Analyze topological indices to identify structural patterns.
    
    Args:
        indices: Dictionary of topological indices
        
    Returns:
        Dictionary containing structural pattern analysis
    """
    analysis = {
        'branching': 'High' if indices['kappa2'] > 2.0 else 'Low',
        'complexity': 'High' if indices['bertz_index'] > 100 else 'Low',
        'connectivity': 'High' if indices['balaban_index'] > 3.0 else 'Low',
        'shape': 'Linear' if indices['kappa1'] < 1.5 else 'Branched'
    }
    
    # Add interpretation
    analysis['interpretation'] = (
        f"This molecule has {analysis['branching']} branching, "
        f"{analysis['complexity']} structural complexity, and "
        f"{analysis['connectivity']} connectivity. "
        f"The overall shape is {analysis['shape']}."
    )
    
    return analysis

def plot_topological_indices_comparison(molecules: List[Dict[str, Any]], 
                                       indices: List[str], 
                                       save_dir: str = DEFAULT_SAVE_DIR) -> None:
    """
    Create bar charts comparing topological indices across molecules.
    
    Args:
        molecules: List of molecule dictionaries
        indices: List of topological index names to compare
        save_dir: Directory to save the plots
    """
    # Prepare data for plotting
    data = {index: [] for index in indices}
    labels = []
    
    for mol in molecules:
        mol_obj = Chem.MolFromSmiles(mol['smiles'])
        if mol_obj is None:
            continue
            
        topo_indices = calculate_topological_indices(mol_obj)
        labels.append(mol['target'][:20] + '...')  # Truncate long labels
        
        for index in indices:
            data[index].append(topo_indices.get(index, 0))
    
    # Create subplots for each index
    n_indices = len(indices)
    fig, axes = plt.subplots(1, n_indices, figsize=(5*n_indices, 5))
    if n_indices == 1:
        axes = [axes]
    
    for ax, index in zip(axes, indices):
        ax.bar(range(len(data[index])), data[index])
        ax.set_title(index)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'topological_indices_comparison.png'))
    plt.close()

def save_molecular_analysis(molecule_data: Dict[str, Any], filename: str, save_dir: str = DEFAULT_SAVE_DIR) -> None:
    """
    Save molecular analysis results to a JSON file.
    
    Args:
        molecule_data: Dictionary containing molecular analysis data
        filename: Name of the file to save
        save_dir: Directory to save the file
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert the dictionary to a serializable format
    serializable_data = {k: convert_to_serializable(v) for k, v in molecule_data.items()}
    
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"Analysis saved to {filepath}")

def main():
    """Main execution function."""
    print("Fetching molecules from ChEMBL...")
    molecules = fetch_chembl_molecules(limit=5)
    
    # Create directory for plots
    os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
    
    # Properties to compare across molecules
    properties_to_compare = ['molecular_weight', 'logp', 'hbd', 'hba']
    
    # Topological indices to compare
    topological_indices = ['wiener_index', 'balaban_index', 'bertz_index', 'randic_index']
    
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
        
        # Calculate and display descriptors
        descriptors = analyze_molecular_descriptors(mol['smiles'])
        print("\nMolecular Descriptors:")
        for key, value in descriptors.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        # Assess CNS drug-likeness
        cns_drug_likeness = assess_cns_drug_likeness(descriptors)
        print(f"\nCNS Drug-Likeness: {cns_drug_likeness['interpretation']}")
        print(f"Score: {cns_drug_likeness['cns_drug_likeness_score']:.2f}")
        
        # Analyze topological patterns
        topo_indices = {k: v for k, v in descriptors.items() if k in calculate_topological_indices(Chem.MolFromSmiles(mol['smiles']))}
        topo_analysis = analyze_topological_patterns(topo_indices)
        print(f"\nTopological Analysis: {topo_analysis['interpretation']}")
        
        # Visualize molecule
        visualize_molecule(mol['smiles'], f"Target: {mol['target']}")
        
        # Save analysis results
        analysis_data = {
            'target': mol['target'],
            'pchembl': mol['pchembl'],
            'smiles': mol['smiles'],
            'properties': props,
            'descriptors': descriptors,
            'cns_assessment': cns_assessment,
            'cns_drug_likeness': cns_drug_likeness,
            'topological_analysis': topo_analysis
        }
        safe_target = "".join(c for c in mol['target'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        save_molecular_analysis(analysis_data, f"{safe_target}_analysis.json")
    
    # Create property comparison plots
    print("\nGenerating property comparison plots...")
    plot_property_comparison(molecules, properties_to_compare)
    
    # Create topological indices comparison plots
    print("\nGenerating topological indices comparison plots...")
    plot_topological_indices_comparison(molecules, topological_indices)
    
    print(f"Plots and analysis saved in '{DEFAULT_SAVE_DIR}' directory")

if __name__ == "__main__":
    main() 