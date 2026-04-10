# Learning Reduced-Round Cipher Behavior Using Machine Learning

## Overview

This project analyzes the learnability of reduced-round lightweight ciphers using machine learning. It tests bitwise accuracy and Hamming distance across different machine learning models (Logistic Regression, MLP, 1D CNN) to identify the round depth where machine learning collapses to chance.

## Project Structure (Code)

| Group | Path | Description |
|---|---|---|
| **Ciphers** | `src/ciphers/` | Reference python implementations of the 16 ciphers. |
| **Core** | `src/dataset.py` | Handles dataset generation, plaintext/ciphertext creation, and XOR masking. |
| | `src/models.py` | PyTorch architectures: Logistic Regression, MLP, and 1D CNN. |
| | `src/train.py` | Training loops, masked BCE loss, and validation checkpointing. |
| | `src/eval.py` | Computes bitwise accuracy and Normalized Hamming Distance. |
| | `src/main.py` | The main experiment runner handling end-to-end grid search. |
| | `src/plot.py` | Generates comparative charts and aggregate exports. |
| **Bonuses** | `bonus/`| Scripts for advanced setups: distinguishing, key generalizability, etc. |

## Ciphers (How They Work)

| Cipher | Type | Internal Structure | Notes |
|---|---|---|---|
| **SIMON** | Block | Feistel | Optimized for hardware; uses bitwise AND, shift, XOR. |
| **SPECK** | Block | ARX | Optimized for software; Addition, Rotation, XOR based. |
| **PRESENT** | Block | SPN | Substitution-Permutation Network, standard lightweight benchmark. |
| **PRINCE** | Block | SPN / FX | Zero-overhead decryption (reflection property). |
| **TEA** | Block | Feistel | Simple algebraic structure, known for simplicity. |
| **XTEA** | Block | Feistel | Key schedule improvement over TEA. |
| **RC5** | Block | Feistel-like | Uses data-dependent rotations. |
| **KATAN** | Block | LFSR | Linear Feedback Shift Register structured. |
| **RECTANGLE** | Block | SPN | Bitslice SPN design, highly efficient in parallel. |
| **CHAM** | Block | ARX | ARX-based, varying rotations depending on round parity. |
| **HIGHT** | Block | ARX | Feistel-like ARX suitable for resource-constrained devices. |
| **LEA** | Block | ARX | Fast execution on general-purpose processors. |
| **SIMECK** | Block | Feistel | Mix of SIMON and SPECK components. |
| **ChaCha20**| Stream | ARX | High-throughput, uses 4x4 matrix state. |
| **Salsa20** | Stream | ARX | Predecessor to ChaCha20. |
| **Trivium** | Stream | NLFSR | Highly compact hardware stream cipher. |

*Stream ciphers are adapted for this project as keystream XOR with plaintext.*

## Bonus Extensions

| Bonus Feature | Script | Description |
|---|---|---|
| **Key Generalization** | `bonus/key_generalization.py` | Tests whether a model trained on one key geometry can predict bits for a different, unseen test key. |
| **Partial Output** | `bonus/partial_output.py` | Narrows down learning target to only one 16-bit word (e.g., Left vs Right in SIMON) to simplify the convergence space. |
| **Distinguisher Setup** | `bonus/distinguisher.py` | Sets up a binary real-vs-random classifier on (PT \|\| CT) rather than doing bitwise prediction. |

## All Results

The table below reports test accuracy (`Test Acc`) and average Hamming distance (`Avg HD`) at variable rounds. **>50% test acc** implies prediction better than random guessing.

| Cipher | Rnd | LR Acc | LR HD | MLP Acc | MLP HD | CNN Acc | CNN HD |
|---|---|---|---|---|---|---|---|
| chacha20 | 1 | 0.4994 | 192.23 | 0.4990 | 192.40 | 0.4985 | 192.56 |
| chacha20 | 2 | 0.4989 | 192.42 | 0.5014 | 191.44 | 0.4996 | 192.17 |
| cham | 1 | 0.5945 | 51.90 | 0.8759 | 15.88 | 0.8786 | 15.54 |
| cham | 2 | 0.5569 | 56.72 | 0.7515 | 31.81 | 0.7555 | 31.30 |
| cham | 3 | 0.5251 | 60.79 | 0.6241 | 48.12 | 0.6344 | 46.80 |
| cham | 4 | 0.5009 | 63.88 | 0.5027 | 63.65 | 0.5094 | 62.80 |
| cham | 5 | 0.4984 | 64.20 | 0.4998 | 64.03 | 0.5054 | 63.31 |
| hight | 1 | 0.5548 | 28.49 | 0.7257 | 17.55 | 0.7531 | 15.80 |
| hight | 2 | 0.4993 | 32.04 | 0.4978 | 32.14 | 0.5044 | 31.72 |
| hight | 3 | 0.5009 | 31.95 | 0.5027 | 31.83 | 0.4966 | 32.22 |
| katan | 1 | 0.6160 | 12.29 | 0.9708 | 0.93 | 0.9790 | 0.67 |
| katan | 2 | 0.5910 | 13.09 | 0.9437 | 1.80 | 0.9571 | 1.37 |
| katan | 3 | 0.6118 | 12.42 | 0.9186 | 2.60 | 0.9370 | 2.02 |
| katan | 4 | 0.5888 | 13.16 | 0.8956 | 3.34 | 0.9180 | 2.62 |
| katan | 5 | 0.5854 | 13.27 | 0.8726 | 4.08 | 0.8962 | 3.32 |
| katan | 6 | 0.5807 | 13.42 | 0.8519 | 4.74 | 0.8769 | 3.94 |
| katan | 7 | 0.5675 | 13.84 | 0.8223 | 5.69 | 0.8536 | 4.68 |
| katan | 8 | 0.5694 | 13.78 | 0.8011 | 6.37 | 0.8297 | 5.45 |
| katan | 9 | 0.5600 | 14.08 | 0.7712 | 7.32 | 0.7960 | 6.53 |
| katan | 10 | 0.5386 | 14.77 | 0.7428 | 8.23 | 0.7699 | 7.36 |
| katan | 11 | 0.5321 | 14.97 | 0.7092 | 9.30 | 0.7438 | 8.20 |
| katan | 12 | 0.5282 | 15.10 | 0.6820 | 10.18 | 0.7118 | 9.22 |
| katan | 13 | 0.5201 | 15.36 | 0.6493 | 11.22 | 0.6868 | 10.02 |
| katan | 14 | 0.5326 | 14.96 | 0.6398 | 11.53 | 0.6658 | 10.69 |
| katan | 15 | 0.5153 | 15.51 | 0.6219 | 12.10 | 0.6418 | 11.46 |
| katan | 16 | 0.5121 | 15.61 | 0.6096 | 12.49 | 0.6280 | 11.90 |
| katan | 17 | 0.5010 | 15.97 | 0.5906 | 13.10 | 0.6057 | 12.62 |
| katan | 18 | 0.5041 | 15.87 | 0.5698 | 13.77 | 0.5874 | 13.20 |
| katan | 19 | 0.5007 | 15.98 | 0.5520 | 14.34 | 0.5708 | 13.73 |
| katan | 20 | 0.4957 | 16.14 | 0.5423 | 14.65 | 0.5597 | 14.09 |
| lea | 1 | 0.5300 | 60.15 | 0.6279 | 47.63 | 0.6276 | 47.67 |
| lea | 2 | 0.5002 | 63.98 | 0.5013 | 63.83 | 0.5014 | 63.83 |
| lea | 3 | 0.5010 | 63.87 | 0.5002 | 63.98 | 0.4977 | 64.29 |
| present | 1 | 0.5521 | 28.67 | 0.6824 | 20.33 | 1.0000 | 0.00 |
| present | 2 | 0.5190 | 30.78 | 0.5289 | 30.15 | 0.6021 | 25.46 |
| present | 3 | 0.4993 | 32.04 | 0.5051 | 31.67 | 0.5115 | 31.27 |
| present | 4 | 0.4986 | 32.09 | 0.4998 | 32.01 | 0.4985 | 32.10 |
| prince | 1 | 0.5002 | 31.99 | 0.5007 | 31.96 | 0.5070 | 31.55 |
| prince | 2 | 0.5059 | 31.62 | 0.4990 | 32.07 | 0.5000 | 32.00 |
| rc5 | 1 | 0.5048 | 31.69 | 0.5097 | 31.38 | 0.5029 | 31.82 |
| rc5 | 2 | 0.4994 | 32.04 | 0.4994 | 32.04 | 0.4996 | 32.03 |
| rectangle | 1 | 0.5171 | 30.91 | 0.6124 | 24.81 | 0.6127 | 24.79 |
| rectangle | 2 | 0.5018 | 31.88 | 0.5075 | 31.52 | 0.5025 | 31.84 |
| rectangle | 3 | 0.5006 | 31.96 | 0.4971 | 32.19 | 0.5002 | 31.98 |
| salsa20 | 1 | 0.5083 | 188.81 | 0.5475 | 173.75 | 0.5439 | 175.13 |
| salsa20 | 2 | 0.4985 | 192.56 | 0.5000 | 192.01 | 0.4994 | 192.24 |
| simeck | 1 | 0.5671 | 13.85 | 0.8423 | 5.05 | 0.8931 | 3.42 |
| simeck | 2 | 0.4987 | 16.04 | 0.5992 | 12.83 | 0.7143 | 9.14 |
| simeck | 3 | 0.5005 | 15.98 | 0.4972 | 16.09 | 0.5062 | 15.80 |
| simeck | 4 | 0.4999 | 16.00 | 0.4996 | 16.01 | 0.4992 | 16.03 |
| simon | 1 | 0.5728 | 13.67 | 0.8435 | 5.01 | 0.8889 | 3.56 |
| simon | 2 | 0.4988 | 16.04 | 0.6009 | 12.77 | 0.6651 | 10.72 |
| simon | 3 | 0.5032 | 15.90 | 0.5057 | 15.82 | 0.5045 | 15.86 |
| simon | 4 | 0.4962 | 16.12 | 0.4972 | 16.09 | 0.4983 | 16.05 |
| speck | 1 | 0.4981 | 16.06 | 0.5638 | 13.96 | 0.6320 | 11.78 |
| speck | 2 | 0.5002 | 15.99 | 0.4991 | 16.03 | 0.4952 | 16.16 |
| speck | 3 | 0.4997 | 16.01 | 0.5025 | 15.92 | 0.4992 | 16.03 |
| tea | 1 | 0.5009 | 31.94 | 0.5012 | 31.92 | 0.4981 | 32.12 |
| tea | 2 | 0.4985 | 32.09 | 0.5027 | 31.82 | 0.4967 | 32.21 |
| trivium | 1 | 0.5009 | 215.59 | 0.4990 | 216.42 | 0.4979 | 216.90 |
| trivium | 2 | 0.4999 | 216.02 | 0.5007 | 215.71 | 0.4989 | 216.46 |
| xtea | 1 | 0.4997 | 32.02 | 0.5011 | 31.93 | 0.4983 | 32.11 |
| xtea | 2 | 0.4991 | 32.06 | 0.4958 | 32.27 | 0.5008 | 31.95 |