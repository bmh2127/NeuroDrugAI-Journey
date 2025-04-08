# Visual Comparison: CNN vs Graph Neural Networks

## CNN Architecture

```
+---+---+---+---+---+---+
|   |   |   |   |   |   |
+---+---+---+---+---+---+
|   |   |   |   |   |   |
+---+---+---+---+---+---+
|   |   |   |   |   |   |
+---+---+---+---+---+---+
|   |   |   |   |   |   |
+---+---+---+---+---+---+
|   |   |   |   |   |   |
+---+---+---+---+---+---+
```

**CNN Input Neurons:**
- Fixed grid structure
- Regular connectivity
- Each neuron represents a pixel
- Uniform number of neighbors

```
+---+---+---+---+
| 1 | 2 | 3 | 4 |
+---+---+---+---+
| 5 | 6 | 7 | 8 |
+---+---+---+---+
| 9 |10 |11 |12 |
+---+---+---+---+
|13 |14 |15 |16 |
+---+---+---+---+
```

**CNN Convolution:**
- Fixed kernel pattern
- Same operation applied everywhere
- Regular spatial relationships

```
+---+---+---+
| K | K | K |
+---+---+---+
| K | K | K |
+---+---+---+
| K | K | K |
+---+---+---+
```

## Graph Neural Network Architecture

```
     A
    / \
   B   C
  / \   \
 D   E   F
      \ /
       G
```

**Graph Nodes:**
- Irregular structure
- Variable connectivity
- Each node represents an atom
- Different numbers of neighbors

```
     A(1)
    /     \
 B(2)     C(3)
  / \       \
D(4) E(5)    F(6)
      \     /
       G(7)
```

**Graph Convolution:**
- Dynamic connectivity
- Message passing between connected nodes
- Preserves topological relationships

```
     A
    / \
   B   C
  / \   \
 D   E   F
      \ /
       G
```

## Key Differences Visualization

### 1. Structure Flexibility

**CNN:**
```
+---+---+---+---+
|   |   |   |   |
+---+---+---+---+
|   |   |   |   |
+---+---+---+---+
|   |   |   |   |
+---+---+---+---+
|   |   |   |   |
+---+---+---+---+
```

**Graph:**
```
     A
    / \
   B   C
  / \   \
 D   E   F
      \ /
       G
```

### 2. Feature Representation

**CNN Input Neuron:**
```
+---+
| 0.7 |  (single value)
+---+
```

**Graph Node:**
```
+-------------------+
| C | 3 | 0 | 1.5 |  (multiple features)
+-------------------+
```

### 3. Information Flow

**CNN:**
```
+---+---+---+---+
|   |   |   |   |
+---+---+---+---+
|   |   |   |   |
+---+---+---+---+
|   |   |   |   |
+---+---+---+---+
|   |   |   |   |
+---+---+---+---+
     ↓
+---+---+---+
|   |   |   |
+---+---+---+  (fixed pattern)
|   |   |   |
+---+---+---+
|   |   |   |
+---+---+---+
```

**Graph:**
```
     A
    / \
   B   C
  / \   \
 D   E   F
      \ /
       G
     ↓
     A
    / \
   B   C
  / \   \
 D   E   F
      \ /
       G
```
(Information flows along edges)

### 4. Neuroscience Parallel

**CNN Input Neurons:**
```
+---+---+---+---+
| R | R | R | R |  (Retina photoreceptors)
+---+---+---+---+
| R | R | R | R |
+---+---+---+---+
| R | R | R | R |
+---+---+---+---+
| R | R | R | R |
+---+---+---+---+
```

**Graph Nodes:**
```
     N
    / \
   N   N
  / \   \
 N   N   N
      \ /
       N
```
(Neurons in brain networks) 