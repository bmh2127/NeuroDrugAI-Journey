from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt

def demonstrate_implicit_hydrogens():
    """
    Demonstrate how hydrogens are implicit in SMILES notation
    by showing structures with and without explicit hydrogens.
    """
    # Create molecules
    ethylamine = Chem.MolFromSmiles('CCN')
    
    # Create a copy with explicit hydrogens
    ethylamine_explicit = Chem.AddHs(ethylamine)
    
    # Create molecules for visualization
    structures = [ethylamine, ethylamine_explicit]
    names = ['Ethylamine (Implicit H)', 'Ethylamine (Explicit H)']
    
    # Add SMILES explanations
    smiles_explanations = [
        'CCN (implicit hydrogens)',
        Chem.MolToSmiles(ethylamine_explicit, allHsExplicit=True)
    ]
    
    # Create a grid of molecule images with explanations
    legends = [f"{name}\n{smile}" for name, smile in zip(names, smiles_explanations)]
    img = Draw.MolsToGridImage(structures, 
                              molsPerRow=2,
                              subImgSize=(300,300),
                              legends=legends,
                              returnPNG=False)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Implicit vs Explicit Hydrogens in SMILES')
    plt.show()
    
    # Print detailed explanation
    print("\nHydrogen Suppression in SMILES:")
    print("1. Carbon (C) in 'CCN':")
    print("   - First C: 1 bond shown → assumed 3 H")
    print("   - Second C: 2 bonds shown → assumed 2 H")
    print("\n2. Nitrogen (N) in 'CCN':")
    print("   - 1 bond shown → assumed 2 H (forming NH2)")
    print("\n3. General Rules:")
    print("   - Carbon: 4 bonds total (shown + implicit H)")
    print("   - Nitrogen: 3 bonds total (shown + implicit H)")
    print("   - Oxygen: 2 bonds total (shown + implicit H)")

if __name__ == "__main__":
    demonstrate_implicit_hydrogens() 