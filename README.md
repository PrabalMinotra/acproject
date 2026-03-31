# Learning Reduced-Round Cipher Behavior using Machine Learning

This repository investigates the boundary condition where machine learning models (Logistic Regression, Multi-Layer Perceptrons, Convolutional Neural Networks) fail to approximate the behavior of 12 distinct lightweight cryptographic block and stream ciphers.

By dynamically reducing the number of encryption rounds ($r \in [1, 5]$) for each cipher, we identify the threshold at which cryptographic diffusion and confusion completely confound standard neural architectures.

## Supported Ciphers (12 Total)
**Block Ciphers**
- SIMON
- SPECK
- PRESENT
- PRINCE
- TEA
- XTEA
- RC5
- KATAN
- RECTANGLE

**Stream Ciphers (adapted by keystream output XORed dynamically)**
- ChaCha20
- Salsa20
- Trivium

Implementation note: 11 ciphers are implemented as validated reference-style variants. PRINCE currently uses a structural reduced-round proxy intended for diffusion-learning experiments rather than a standards-validated drop-in implementation.

## Project Structure

- `src/ciphers/`: Contains the 12 variable-round, bare-metal Python cipher implementations.
- `src/dataset.py`: PyTorch dataset generator scaling dynamically for all block bit widths ($32$ to $512$ bit). 
- `src/models.py`: Definitions for Logistic Regression, MLP, and 1D CNN models supporting variable input blocks.
- `src/train.py`: The PyTorch training loop saving model checkpoints for the best validation accuracy.
- `src/eval.py`: Calculates classification bitwise accuracy and normalized Hamming distance versus the true ciphertext labels.
- `src/main.py`: The master orchestration script running the permutation pipeline (r=[1..5]) across all 12 ciphers concurrently.
- `src/plot.py`: Generates aggregate comparative visualizations.
- `results/`: Directory containing all serialized model `.pt` files, metrics JSON, and generated plots.
- `report/`: A `.tex` academic report documenting the experimental findings and diffusion metrics.

## Quickstart Guide

### 1. Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Data and Train Models
Run the massive execution pipeline to evaluate the 12 ciphers concurrently across all rounds and models. Data generation and prediction metrics will be continually synchronized to `results/metrics.json`. Wait for it to complete.

```bash
export PYTHONPATH=.
python3 src/main.py
```

### 3. Generate Comparative Plots
Once the `results/metrics.json` is completely populated, generate the cross-cipher visualizations.
```bash
python3 src/plot.py
```

The comprehensive line plots isolating the precise moment ML model degradation hits exactly 50% random guessing accuracy per cipher will be saved to `results/plots/`.

### 4. Compile Report (Optional)
Requires a full LaTeX distribution (like MacTeX or TeX Live).
```bash
cd report/
pdflatex report.tex
```
