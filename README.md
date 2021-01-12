# Lmser-DCW
Project of CS7319 [2020 Fall]

Proceeded beyond Auto Encoder (AE), Lmser is featured by a bidirectional architecture with several built-in natures, for which you can refer to the table below:  
![image](https://github.com/MatthewXY01/Lmser-DCW/blob/main/misc/duality.png)

In this project, we implement one major nature of Lmser: Duality in Connection Weights (DCW). DCW refers to using the symmetric weights in corresponding layers in encoder and decoder such that the direct cascading by AE is further enhanced.

The purpose of designing such a symmetrical structure in Lmser is to approximate identity mapping per two consecutive layers simply through *W'W≈I*.

We have also implemented another version of DCW with constraint of Moore-Penrose pseudoinverse.

## Dataset

MNIST: http://yann.lecun.com/exdb/mnist/

F-MNIST: https://github.com/zalandoresearch/fashion-mnist

## Run

for vanilla AutoEncoder:

```python
python run.py --model AE
```

for DCW without pseudoinverse-constraint:

```python
python run.py --model DCW_woConstraint
```

for DCW with pseudoinverse-constraint:

```python
python run.py --model DCW
```



## Requirement

- PyTorch