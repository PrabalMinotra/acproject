# Learning Reduced-Round Cipher Behavior Using Machine Learning

This project analyzes how well common machine learning models can learn reduced-round
lightweight ciphers, and identifies the round depth where learning collapses to chance.
It runs controlled experiments across 16 ciphers and records bitwise accuracy and
normalized Hamming distance as rounds increase.

Supported ciphers (16 total)

Block ciphers

- SIMON
- SPECK
- PRESENT
- PRINCE
- TEA
- XTEA
- RC5
- KATAN
- RECTANGLE
- CHAM
- HIGHT
- LEA
- SIMECK

Stream ciphers (adapted as keystream XOR with plaintext)

- ChaCha20
- Salsa20
- Trivium

Implementation note: PRINCE is implemented as a reduced-round structural proxy for
diffusion-learning experiments rather than a standards-validated drop-in variant.

Project layout

- src/ciphers/: reference-style cipher implementations with variable rounds
- src/dataset.py: plaintext/ciphertext dataset generation and masking logic
- src/models.py: Logistic Regression, MLP, and 1D CNN models
- src/train.py: training loop with masked BCE loss and checkpointing
- src/eval.py: bitwise accuracy and average Hamming distance evaluation
- src/main.py: main experiment runner across ciphers/rounds/models
- src/plot.py: comparative plots and summary exports
- results/: checkpoints, logs, metrics, and plots
- report/: LaTeX report
- bonus/: optional extensions described below

Results and artifacts

- results/metrics.json: per-cipher metrics by round and model
- results/models/: model checkpoints
- results/logs/: training logs
- results/plots/: aggregate figures

Bonus extensions

- Key generalization: train on one key, test on another (defaults to SIMON and KATAN)
- Partial output learning: predict only one 16-bit word of SIMON32/64 (left and right)
- Distinguisher setup: binary real-vs-random classifier on (PT || CT)

Bonus outputs are written to bonus/results and bonus data caches live in bonus/data.

Step-by-step run (PowerShell)

```
python -m venv venv
.\.venv\Scripts\Activate.ps1
python -m src.main
python -m src.plot
python -m bonus.key_generalization
python -m bonus.distinguisher
python -m bonus.partial_output
deactivate
```

Notes

- The bonus scripts are isolated from the main results; clearing bonus/data is
  enough to refresh bonus datasets.
