# Pruning + Quantization on Fashion-MNIST

## Project Title
**“Half the Weights, Half the Bits”**  
An illustrative example of combining **global unstructured pruning** and **dynamic post-training quantization** to shrink a CNN while retaining reasonable accuracy.

## Description
We train a small convolutional network on Fashion-MNIST, then:

1. Globally prune 50 % of all convolutional and linear weights by L1 magnitude.  
2. Apply dynamic quantization (INT8) to the remaining linear layers.  

The script outputs accuracy and on-disk size for:

| Model | Compression | Notes |
|-------|-------------|-------|
| Baseline | _None_ | FP32 |
| Pruned + Quantized | **50 % sparse + INT8** | linear layers quantized |

## Dataset
- **Fashion-MNIST** (10 classes, 28×28 grayscale)  

## sample output
```bash
=== Compression Results ===
            Model  Accuracy  Size_MB
          Baseline     0.892      0.39
 Pruned+Quantized     0.876      0.15

Finished in 175.4 s
```

## Examples
```bash
# Change pruning ratio to 80 %
sed -i 's/amount=0.5/amount=0.8/' main.py
python main.py
````
