#!/usr/bin/env python3
"""
Molecular Featurization Examples
Demonstrates different molecular featurization techniques for CNS drug discovery.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import deepchem as dc
from deepchem.feat import ConvMolFeaturizer, WeaveFeaturizer, MolGraphConvFeaturizer
from deepchem.feat.mol_graphs import ConvMol
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_conformer(mol: Chem.Mol, num_confs: int = 10) -> Chem.Mol:
    """
    Generate 3D conformers for a molecule.
    
    Args:
        mol: RDKit molecule object
        num_confs: Number of conformers to generate
        
    Returns:
        Molecule with embedded conformers
    """
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate conformers
    AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=42)
    
    # Optimize conformers
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    
    return mol

def calculate_coulomb_matrix(mol: Chem.Mol, conf_id: int = 0) -> np.ndarray:
    """
    Calculate Coulomb matrix for a molecule conformer.
    
    Args:
        mol: RDKit molecule object with conformers
        conf_id: Conformer ID to use
        
    Returns:
        Coulomb matrix as numpy array
    """
    # Get atomic numbers and coordinates
    conf = mol.GetConformer(conf_id)
    num_atoms = mol.GetNumAtoms()
    
    # Initialize matrix
    cm = np.zeros((num_atoms, num_atoms))
    
    # Calculate matrix elements
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                # Diagonal elements: 0.5 * Z_i^2.4
                cm[i,j] = 0.5 * mol.GetAtomWithIdx(i).GetAtomicNum()**2.4
            else:
                # Off-diagonal elements: Z_i * Z_j / |R_i - R_j|
                pos_i = conf.GetAtomPosition(i)
                pos_j = conf.GetAtomPosition(j)
                dist = np.sqrt((pos_i.x - pos_j.x)**2 + 
                             (pos_i.y - pos_j.y)**2 + 
                             (pos_i.z - pos_j.z)**2)
                if dist > 0:
                    cm[i,j] = (mol.GetAtomWithIdx(i).GetAtomicNum() * 
                             mol.GetAtomWithIdx(j).GetAtomicNum() / dist)
    
    return cm

def compare_featurizers(smiles: str) -> Dict[str, Any]:
    """
    Compare different molecular featurization techniques.
    
    Args:
        smiles: SMILES string of the molecule
        
    Returns:
        Dictionary containing different featurizations
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # Generate 3D conformers
    mol_3d = generate_conformer(mol)
    
    # Calculate Coulomb matrix
    cm = calculate_coulomb_matrix(mol_3d)
    
    # Generate different featurizations
    conv_mol_feat = ConvMolFeaturizer()
    weave_feat = WeaveFeaturizer()
    graph_conv_feat = MolGraphConvFeaturizer()
    
    # Apply featurizers
    conv_mol = conv_mol_feat.featurize([mol])[0]
    weave = weave_feat.featurize([mol])[0]
    graph_conv = graph_conv_feat.featurize([mol])[0]
    
    return {
        'coulomb_matrix': cm,
        'conv_mol': conv_mol,
        'weave': weave,
        'graph_conv': graph_conv
    }

def visualize_featurizations(featurizations: Dict[str, Any], 
                           save_dir: str = "featurization_plots") -> None:
    """
    Visualize different molecular featurizations.
    
    Args:
        featurizations: Dictionary of featurizations
        save_dir: Directory to save visualizations
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Plot Coulomb matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(featurizations['coulomb_matrix'], 
                    cmap='viridis', 
                    square=True)
        plt.title('Coulomb Matrix')
        plt.savefig(os.path.join(save_dir, 'coulomb_matrix.png'))
        plt.close()
        
        # Plot ConvMol features
        plt.figure(figsize=(8, 6))
        plt.imshow(featurizations['conv_mol'].get_atom_features(), 
                   aspect='auto', 
                   cmap='viridis')
        plt.title('ConvMol Atom Features')
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, 'conv_mol_features.png'))
        plt.close()
        
        # Plot Weave features
        plt.figure(figsize=(8, 6))
        plt.imshow(featurizations['weave'].get_atom_features(), 
                   aspect='auto', 
                   cmap='viridis')
        plt.title('Weave Atom Features')
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, 'weave_features.png'))
        plt.close()
        
        # Plot GraphConv node features
        plt.figure(figsize=(8, 6))
        # GraphData object has node_features instead of get_atom_features()
        plt.imshow(featurizations['graph_conv'].node_features, 
                   aspect='auto', 
                   cmap='viridis')
        plt.title('GraphConv Node Features')
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, 'graph_conv_features.png'))
        plt.close()
        
        # Handle GraphConv edge features - they might be in a different format
        try:
            # Try to convert edge features to numpy array if needed
            edge_features = featurizations['graph_conv'].edge_features
            if not isinstance(edge_features, np.ndarray):
                # If it's a list or other format, try to convert it
                edge_features = np.array(edge_features, dtype=float)
            
            # Only plot if we have valid edge features
            if edge_features.size > 0 and edge_features.ndim == 2:
                plt.figure(figsize=(8, 6))
                plt.imshow(edge_features, 
                           aspect='auto', 
                           cmap='viridis')
                plt.title('GraphConv Edge Features')
                plt.colorbar()
                plt.savefig(os.path.join(save_dir, 'graph_conv_edge_features.png'))
                plt.close()
            else:
                print("Warning: Edge features are not in a format suitable for visualization")
        except Exception as e:
            print(f"Warning: Could not visualize edge features: {str(e)}")
        
        # Save feature dimensions for reference
        with open(os.path.join(save_dir, 'feature_dimensions.txt'), 'w') as f:
            f.write("Feature Dimensions:\n")
            f.write(f"Coulomb Matrix: {featurizations['coulomb_matrix'].shape}\n")
            f.write(f"ConvMol Features: {featurizations['conv_mol'].get_atom_features().shape}\n")
            f.write(f"Weave Features: {featurizations['weave'].get_atom_features().shape}\n")
            f.write(f"GraphConv Node Features: {featurizations['graph_conv'].node_features.shape}\n")
            
            # Try to get edge feature dimensions safely
            try:
                edge_features = featurizations['graph_conv'].edge_features
                if isinstance(edge_features, np.ndarray):
                    f.write(f"GraphConv Edge Features: {edge_features.shape}\n")
                else:
                    f.write(f"GraphConv Edge Features: {type(edge_features)}\n")
            except:
                f.write("GraphConv Edge Features: Not available\n")
    
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        print("Saving feature information to text file instead...")
        
        # Save basic information about features
        with open(os.path.join(save_dir, 'feature_info.txt'), 'w') as f:
            f.write("Feature Information:\n")
            for key, value in featurizations.items():
                f.write(f"{key}: {type(value)}\n")
                if hasattr(value, 'shape'):
                    f.write(f"  Shape: {value.shape}\n")
                elif hasattr(value, '__len__'):
                    f.write(f"  Length: {len(value)}\n")

def main():
    """Main execution function."""
    # Example CNS drug molecule (dopamine)
    smiles = "C(C1=CC(=C(C=C1)O)O)CN"
    
    print("Comparing featurization techniques for dopamine...")
    featurizations = compare_featurizers(smiles)
    
    print("Visualizing featurizations...")
    visualize_featurizations(featurizations)
    
    print("Analysis complete. Visualizations saved in 'featurization_plots' directory")

if __name__ == "__main__":
    main() 