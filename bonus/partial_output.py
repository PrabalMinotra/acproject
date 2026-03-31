import argparse
import json
import os
import time

import torch

from src.dataset import get_dataloaders
from src.models import LogisticRegressionModel, MLPModel, CNNModel
from src.train import train_model
from src.eval import evaluate_model

MODELS = {
    'LogisticRegression': LogisticRegressionModel,
    'MLP': MLPModel,
    'CNN': CNNModel,
}


def parse_hex_key(value):
    if value is None:
        return None
    text = value.strip().lower()
    if text.startswith('0x'):
        text = text[2:]
    if not text:
        raise ValueError('Empty hex key')
    return int(text, 16)


def parse_rounds(text):
    text = text.strip()
    if '-' in text:
        start, end = text.split('-', 1)
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in text.split(',') if x.strip()]


def parse_words(word):
    if word == 'both':
        return ['left', 'right']
    return [word]


def detect_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def build_word_mask(block_size, word, word_bits):
    mask = torch.zeros(block_size, dtype=torch.float32)
    if word == 'left':
        mask[:word_bits] = 1.0
    else:
        mask[block_size - word_bits:] = 1.0
    return mask


def extract_base_mask(loader):
    dataset = loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    return dataset.label_mask_tensor.clone().float()


def apply_mask_to_loader(loader, mask_tensor):
    dataset = loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    dataset.label_mask_tensor = mask_tensor
    dataset.label_mask = mask_tensor.cpu().numpy()


def run_partial_output(
    cipher_name,
    rounds,
    key,
    words,
    num_samples,
    epochs,
    batch_size,
    cache_dir,
    results_path,
):
    device = detect_device()
    print(f"Using device: {device}")

    results_dir = os.path.dirname(results_path)
    models_dir = os.path.join(results_dir, 'models')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    all_results = {cipher_name: {}}

    for r in rounds:
        train_loader, val_loader, test_loader, block_size, _ = get_dataloaders(
            num_samples=num_samples,
            rounds=r,
            key=key,
            cipher_name=cipher_name,
            batch_size=batch_size,
            cache_dir=cache_dir,
        )

        word_bits = block_size // 2
        base_mask = extract_base_mask(train_loader)
        round_entry = all_results[cipher_name].setdefault(str(r), {})

        for word in words:
            word_mask = build_word_mask(block_size, word, word_bits)
            mask_tensor = base_mask * word_mask

            apply_mask_to_loader(train_loader, mask_tensor)
            apply_mask_to_loader(val_loader, mask_tensor)
            apply_mask_to_loader(test_loader, mask_tensor)

            output_bits = int(mask_tensor.sum().item())
            word_entry = round_entry.setdefault(word, {})

            for model_name, model_cls in MODELS.items():
                model = model_cls(input_size=block_size, output_size=block_size)
                save_path = os.path.join(models_dir, f"{cipher_name}_{model_name}_r{r}_{word}.pt")

                start_time = time.time()
                history = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    device=device,
                    save_path=save_path,
                )

                if os.path.exists(save_path):
                    model.load_state_dict(torch.load(save_path, weights_only=True))

                test_acc, avg_hamming = evaluate_model(model, test_loader, device=device)
                elapsed = time.time() - start_time

                print(
                    f"[r={r} {word} {model_name}] Test Acc: {test_acc:.4f} | "
                    f"Avg Hamming: {avg_hamming:.2f} | Time: {elapsed:.1f}s"
                )

                word_entry[model_name] = {
                    'test_acc': test_acc,
                    'avg_hamming': avg_hamming,
                    'history': history,
                    'time_s': elapsed,
                    'block_size': block_size,
                    'output_bits': output_bits,
                    'word': word,
                }

                with open(results_path, 'w', encoding='ascii') as f:
                    json.dump(all_results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn only part of the output (one word).')
    parser.add_argument('--cipher', type=str, default='simon')
    parser.add_argument('--rounds', type=str, default='1,2')
    parser.add_argument('--key', type=str, default=None)
    parser.add_argument('--word', choices=['left', 'right', 'both'], default='both')
    parser.add_argument('--num-samples', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--cache-dir', type=str, default='bonus/data/partial_output')
    parser.add_argument('--results', type=str, default='bonus/results/partial_output.json')
    args = parser.parse_args()

    run_partial_output(
        cipher_name=args.cipher.lower(),
        rounds=parse_rounds(args.rounds),
        key=parse_hex_key(args.key) if args.key else 0x1918111009080100,
        words=parse_words(args.word),
        num_samples=args.num_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        results_path=args.results,
    )
