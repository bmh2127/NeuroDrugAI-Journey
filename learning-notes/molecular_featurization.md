# Molecular Featurization Techniques

## Overview
Molecular featurization is a critical step in machine learning for drug discovery, transforming molecular structures into numerical representations that can be processed by ML algorithms. This note covers various featurization techniques available in DeepChem and their applications.

## Key Featurization Methods

### 1. RDKitDescriptors
- Computes basic physical and chemical properties using RDKit
- Includes properties like:
  - Molecular weight
  - Polar surface area
  - Number of hydrogen bond donors/acceptors
  - QED (Quantitative Estimate of Drug-likeness)
- Best for predicting properties that depend on high-level molecular characteristics
- Provides ~210 different descriptors

### 2. Graph-Based Featurizers
- **ConvMolFeaturizer**: Converts molecules into ConvMol objects for graph convolutions
- **WeaveFeaturizer**: Alternative graph representation for specific models
- **MolGraphConvFeaturizer**: Another graph-based featurization approach
- These methods preserve molecular structure information through graph representations
- Particularly useful for tasks requiring structural understanding

### 3. Coulomb Matrix
- Represents molecular conformations through electrostatic interactions
- Creates an N×N matrix where elements represent electrostatic interactions between atom pairs
- Key features:
  - Invariant to molecular rotation and translation
  - Captures both atomic charges and interatomic distances
  - Requires conformer generation for flexible molecules
- Limitations:
  - Not invariant to atom index permutations
  - Matrix size varies with number of atoms (requires padding)

### 4. Coulomb Matrix Eigenvalues
- Derived from Coulomb matrix but uses eigenvalue spectrum
- Advantages:
  - Invariant to both rotation/translation AND atom index permutations
  - More compact representation (N eigenvalues vs N×N matrix)
- Disadvantages:
  - Contains less information than full Coulomb matrix
  - May limit model learning capacity

### 5. SMILES Tokenization
- Breaks SMILES strings into tokens for sequence models
- Process:
  1. Tokenization: Break SMILES into substrings
  2. Numericalization: Convert tokens to integers
  3. Zero-padding: Ensure consistent sequence lengths
- Enables use of sequence models (CNNs, transformers)
- Allows models to learn their own molecular representations

## Implementation Considerations

### Conformer Generation
- Required for 3D-based featurizations (Coulomb Matrix)
- Use `ConformerGenerator` class in DeepChem
- Parameters:
  - `max_conformers`: Limit number of conformers
  - Energy minimization and pruning for distinct conformers

### Data Processing Pipeline
1. Choose appropriate featurizer based on task
2. Generate conformers if needed
3. Apply featurization
4. Handle variable-sized inputs (padding/truncation)
5. Create dataset compatible with chosen ML model

## Best Practices

1. **Task-Specific Selection**
   - Use RDKitDescriptors for property prediction
   - Graph-based methods for structure-dependent tasks
   - Coulomb matrices for 3D-dependent properties
   - SMILES tokenization for sequence-based learning

2. **Data Preparation**
   - Ensure consistent input sizes
   - Handle missing or invalid molecules
   - Consider computational requirements

3. **Model Compatibility**
   - Match featurization to model architecture
   - Consider invariance requirements
   - Balance information content vs computational cost

## References
- DeepChem Tutorial: "Going Deeper on Molecular Featurizations"
- RDKit Documentation
- DeepChem Documentation 