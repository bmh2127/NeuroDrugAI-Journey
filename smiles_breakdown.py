from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt

def demonstrate_dopamine_breakdown():
    """
    Demonstrate how dopamine's structure is represented in SMILES notation
    by highlighting different structural components.
    """
    # Full dopamine structure
    dopamine = Chem.MolFromSmiles('C(C1=CC(=C(C=C1)O)O)CN')
    
    # Breaking down into substructures
    aromatic_ring = Chem.MolFromSmiles('C1=CC=CC=C1')  # Basic benzene ring
    catechol = Chem.MolFromSmiles('OC1=CC=C(O)C=C1')   # Ring with diols
    ethylamine = Chem.MolFromSmiles('CCN')             # Ethylamine chain
    
    # Create labels for each structure
    structures = [dopamine, aromatic_ring, catechol, ethylamine]
    names = ['Complete Dopamine', 
             'Aromatic Ring Core', 
             'Catechol Group',
             'Ethylamine Chain']
    
    # Add detailed SMILES explanations
    smiles_explanations = [
        'C(C1=CC(=C(C=C1)O)O)CN',
        'C1=CC=CC=C1',
        'OC1=CC=C(O)C=C1',
        'CCN'
    ]
    
    # Create a grid of molecule images with explanations
    legends = [f"{name}\n{smile}" for name, smile in zip(names, smiles_explanations)]
    img = Draw.MolsToGridImage(structures, 
                              molsPerRow=2,
                              subImgSize=(300,300),
                              legends=legends,
                              returnPNG=False)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Dopamine SMILES Notation Breakdown')
    plt.show()
    
    # Print detailed explanation
    print("\nDopamine SMILES Breakdown:")
    print("1. Aromatic Ring: C1=CC=CC=C1")
    print("   - Numbers (1) mark ring closure points")
    print("   - =C represents double bonds in alternating pattern")
    print("\n2. Catechol Group: OC1=CC=C(O)C=C1")
    print("   - O represents hydroxyl groups")
    print("   - Parentheses show branching points")
    print("\n3. Ethylamine Chain: CCN")
    print("   - Simple chain of two carbons")
    print("   - N represents terminal amine group")

if __name__ == "__main__":
    demonstrate_dopamine_breakdown() 