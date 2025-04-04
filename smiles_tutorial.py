from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

def demonstrate_smiles_basics():
    """
    Demonstrate basic SMILES notation concepts using CNS-relevant molecules.
    """
    # Example 1: Basic atom representation
    # Dopamine: C(C1=CC(=C(C=C1)O)O)CN
    dopamine = Chem.MolFromSmiles('C(C1=CC(=C(C=C1)O)O)CN')
    
    # Example 2: Bond types
    # Serotonin (5-HT): NCCc1c[nH]c2ccc(O)cc12
    serotonin = Chem.MolFromSmiles('NCCc1c[nH]c2ccc(O)cc12')
    
    # Example 3: Ring structures
    # Diazepam (Valium): CC1=CC=C(C=C1)C2=NC(C(=O)NC3=C2C=CC=C3)(C)C
    diazepam = Chem.MolFromSmiles('CC1=CC=C(C=C1)C2=NC(C(=O)NC3=C2C=CC=C3)(C)C')
    
    # Visualize molecules
    molecules = [dopamine, serotonin, diazepam]
    names = ['Dopamine', 'Serotonin', 'Diazepam']
    
    # Create a grid of molecule images
    img = Draw.MolsToGridImage(molecules, 
                              molsPerRow=3,
                              subImgSize=(300,300),
                              legends=names,
                              returnPNG=False)
    
    plt.figure(figsize=(15, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.title('SMILES Examples: CNS-Relevant Molecules')
    plt.show()
    
    # Demonstrate SMILES to structure conversion
    print("\nSMILES to Structure Examples:")
    for mol, name in zip(molecules, names):
        print(f"\n{name}:")
        print(f"SMILES: {Chem.MolToSmiles(mol)}")
        print(f"Number of atoms: {mol.GetNumAtoms()}")
        print(f"Number of bonds: {mol.GetNumBonds()}")

if __name__ == "__main__":
    demonstrate_smiles_basics() 