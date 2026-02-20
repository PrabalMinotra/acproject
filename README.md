# Learning Reduced-Round Cipher Behavior using Machine Learning
**Author:** Prabal Minotra / Academic Project
**Topics:** Cryptanalysis, SIMON Block Cipher, PyTorch, Deep Learning

This project analyzes whether Machine Learning models (Logistic Regression, Multi-Layer Perceptrons, Convolutional Neural Networks) can effectively learn the input-output mappings of the SIMON32/64 lightweight block cipher. We explore learning degradation as the number of encryption rounds ($r$) increases.

## Structure
- `src/simon.py`: A clean, verifiable implementation of the SIMON32/64 cipher optimized for configurable limited-round encryption.
- `src/dataset.py`: PyTorch `Dataset` framework that generates 100,000 mapping pairs of random Plaintexts to Ciphertexts for $r$ rounds. 
- `src/models.py`: Modular definitions of three Neural Network structures.
- `src/train.py` & `src/eval.py`: Boilerplate loops for training the networks with Adam and BinaryCrossEntropy, and tracking metrics like Average Hamming Distance.
- `src/main.py`: The entry-point orchestrator script executing training epochs across rounds 1 through 5.
- `src/plot.py`: Takes the `results/metrics.json` outputs and creates publication-ready pyplot graphs.
- `results/`: Contains saved `.pt` models, JSON metrics, and the output `plots`.
- `report/`: Contains the IEEE format academic report detailing the project.

## How To Run
The project depends on `torch`, `numpy`, `matplotlib`, and `tqdm`. Running via `venv` on a Mac or Linux machine is recommended.

1. **Setup the Virtual Environment & Dependencies:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Execute the Experiment Pipeline:**
The main pipeline iterates through $r \in [1, 5]$, trains three distinct models, aggregates all evaluations, and saves them to `results/metrics.json`.
```bash
# Optional: Ensure the working directory is configured in PYTHONPATH
PYTHONPATH=. python3 src/main.py
```

3. **Generate Plots:**
Uses the JSON artifact to produce plots visualising learning degradation.
```bash
PYTHONPATH=. python3 src/plot.py
```

## Results Overview
The models learn 1-round efficiently. Deeper models (MLP, CNN) sustain functionality through round 2 and partially round 3. By round 4, the SIMON permutation becomes universally unlearnable under our constraints, representing the cipher's rapid diffusion wall where the models degenerate to random guessing.
