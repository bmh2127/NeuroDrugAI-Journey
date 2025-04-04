#!/usr/bin/env python3
"""
ECFP Visualization Script
This script demonstrates the Extended Connectivity Fingerprint (ECFP) generation process
using RDKit, with visualizations of each iteration.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt
import os
from pathlib import Path
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_ecfp_visualization(mol: Chem.Mol, radius: int = 2, save_dir: str = "ecfp_plots") -> None:
    """
    Create visualization of ECFP generation process.
    
    Args:
        mol: RDKit molecule object
        radius: ECFP radius (number of iterations)
        save_dir: Directory to save visualizations
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)
    
    # Create figure with subplots for each iteration
    fig, axes = plt.subplots(radius + 1, 1, figsize=(10, 4*(radius + 1)))
    if radius == 0:
        axes = [axes]
    
    # Plot original molecule
    img = Draw.MolToImage(mol)
    axes[0].imshow(img)
    axes[0].set_title("Original Molecule")
    axes[0].axis('off')
    
    # Generate and visualize ECFP iterations
    for i in range(radius):
        # Generate ECFP for this iteration
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, i + 1)
        
        # Create a copy of the molecule for highlighting
        highlight_mol = Chem.Mol(mol)
        
        # Get atom indices for this iteration
        info = {}
        AllChem.GetMorganFingerprint(mol, i + 1, bitInfo=info)
        
        # Collect all atoms to highlight
        highlight_atoms = []
        for atom_indices in info.values():
            for atom_idx_tuple in atom_indices:
                highlight_atoms.extend(atom_idx_tuple)
        
        # Draw molecule with highlights - using tuple for highlightColor (R,G,B)
        img = Draw.MolToImage(highlight_mol, highlightAtoms=highlight_atoms,
                            highlightColor=(1, 0, 0))  # Red as RGB tuple
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"ECFP Iteration {i + 1}\nFeatures: {len(info)}")
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ecfp_process.png'))
    plt.close()

def visualize_ecfp_features(mol: Chem.Mol, radius: int = 2, save_dir: str = "ecfp_plots") -> None:
    """
    Visualize individual ECFP features.
    
    Args:
        mol: RDKit molecule object
        radius: ECFP radius
        save_dir: Directory to save visualizations
    """
    # Generate ECFP information
    info = {}
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    
    # Create a figure for each feature
    for bit, atom_indices in info.items():
        # Create a copy of the molecule
        highlight_mol = Chem.Mol(mol)
        
        # Process atom indices to handle potential nested structure
        highlight_atoms = []
        for atom_idx_tuple in atom_indices:
            if isinstance(atom_idx_tuple, tuple):
                highlight_atoms.extend(atom_idx_tuple)
            else:
                highlight_atoms.append(atom_idx_tuple)
        
        # Draw molecule with highlights - using tuple for highlightColor (R,G,B)
        img = Draw.MolToImage(highlight_mol, highlightAtoms=highlight_atoms,
                            highlightColor=(1, 0, 0))  # Red as RGB tuple
        
        # Create and save figure
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Feature {bit}")
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'feature_{bit}.png'))
        plt.close()

def demonstrate_feature_uniqueness(mol: Chem.Mol, radius: int = 2, save_dir: str = "ecfp_plots") -> None:
    """
    Demonstrate how ECFP features are uniquely identified based on structural patterns.
    
    Args:
        mol: RDKit molecule object
        radius: ECFP radius
        save_dir: Directory to save visualizations
    """
    # Generate ECFP information
    info = {}
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    
    # Create a figure showing feature uniqueness
    fig = plt.figure(figsize=(15, 15))
    
    # Select four different features to demonstrate
    feature_bits = list(info.keys())[:4]
    
    # Create a text file for descriptions
    desc_file = os.path.join(save_dir, 'feature_descriptions.txt')
    
    with open(desc_file, 'w') as f:
        f.write("ECFP Feature Descriptions\n")
        f.write("=======================\n\n")
        
        for idx, bit in enumerate(feature_bits):
            # Create a copy of the molecule
            highlight_mol = Chem.Mol(mol)
            
            # Get atom indices for this feature
            atom_indices = info[bit]
            
            # Process atom indices to handle potential nested structure
            highlight_atoms = []
            for atom_idx_tuple in atom_indices:
                if isinstance(atom_idx_tuple, tuple):
                    highlight_atoms.extend(atom_idx_tuple)
                else:
                    highlight_atoms.append(atom_idx_tuple)
            
            # Draw molecule with highlights
            img = Draw.MolToImage(highlight_mol, highlightAtoms=highlight_atoms,
                                highlightColor=(1, 0, 0))
            
            # Get feature description
            feature_desc = get_feature_description(mol, highlight_atoms[0], radius)
            
            # Create subplot
            ax = plt.subplot(2, 2, idx + 1)
            
            # Display molecule image
            ax.imshow(img)
            ax.axis('off')
            
            # Add title with feature number
            ax.set_title(f"Feature {bit}", pad=20, fontsize=12)
            
            # Write description to file
            f.write(f"Feature {bit}\n")
            f.write("-" * 50 + "\n")
            f.write(feature_desc + "\n\n")
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_uniqueness.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def get_feature_description(mol: Chem.Mol, atom_idx: int, radius: int) -> str:
    """
    Generate a human-readable description of a feature using OpenAI API.
    
    Args:
        mol: RDKit molecule object
        atom_idx: Index of the central atom
        radius: ECFP radius
        
    Returns:
        String description of the feature
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    desc = []
    
    # Get basic chemical information
    central_symbol = atom.GetSymbol()
    formal_charge = atom.GetFormalCharge()
    num_explicit_hs = atom.GetNumExplicitHs()
    num_implicit_hs = atom.GetNumImplicitHs()
    
    # Get bond information
    bonds = []
    for bond in atom.GetBonds():
        neighbor = bond.GetOtherAtom(atom)
        bond_type = bond.GetBondType()
        neighbor_symbol = neighbor.GetSymbol()
        bonds.append(f"{bond_type} to {neighbor_symbol}")
    
    # Create a prompt for OpenAI
    prompt = f"""In the context of molecular fingerprints and drug discovery, explain what makes this chemical feature unique:
    - Central atom: {central_symbol}
    - Formal charge: {formal_charge}
    - Explicit hydrogens: {num_explicit_hs}
    - Implicit hydrogens: {num_implicit_hs}
    - Bonds: {', '.join(bonds)}
    - Environment radius: {radius}
    
    Focus on:
    1. The structural uniqueness of this feature
    2. Why this pattern is important for molecular recognition
    3. How this might relate to biological activity
    4. Any special chemical properties or reactivity
    
    Keep the response concise and technical."""
    
    try:
        # Get API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Generate description using OpenAI
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in chemical informatics and drug discovery, specializing in molecular fingerprints and structure-activity relationships."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        # Extract the generated description
        ai_description = response.choices[0].message.content.strip()
        
        # Combine basic info with AI description
        desc.append(f"Basic Information:")
        desc.append(f"- Central atom: {central_symbol}")
        desc.append(f"- Formal charge: {formal_charge}")
        desc.append(f"- Bonds: {', '.join(bonds)}")
        desc.append(f"\nAI-Generated Description:")
        desc.append(ai_description)
        
    except Exception as e:
        # Fallback to basic description if API call fails
        desc.append(f"Central atom: {central_symbol}")
        desc.append("Bonds: " + ", ".join(bonds))
        desc.append(f"Environment radius: {radius}")
        desc.append(f"\nError generating AI description: {str(e)}")
    
    return "\n".join(desc)

def main():
    """Main execution function."""
    # Example molecule (caffeine)
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        print("Could not parse SMILES string")
        return
    
    # Create visualizations
    print("Generating ECFP process visualization...")
    create_ecfp_visualization(mol, radius=2)
    
    print("Generating individual feature visualizations...")
    visualize_ecfp_features(mol, radius=2)
    
    print("Demonstrating feature uniqueness...")
    demonstrate_feature_uniqueness(mol, radius=2)
    
    print("Visualizations saved in 'ecfp_plots' directory")

if __name__ == "__main__":
    main()