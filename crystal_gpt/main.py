import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import GPTConfig, TrainerConfig, GPT, generate_target

import argparse

# Dataset class
class LargeTextDataset(Dataset):
    def __init__(self, data_list, ctoi, block_size):
        self.data_list = data_list
        self.ctoi = ctoi
        self.block_size = block_size

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        encoding = data['encoding']
        target = data['energy']  # or 'area'

        combined_text = f"{encoding}\n{target}\n"
        encoded_text = [self.ctoi.get(c, self.ctoi['<unk>']) for c in combined_text]

        # Truncate or pad
        if len(encoded_text) > self.block_size:
            encoded_text = encoded_text[:self.block_size]
        else:
            padding_length = self.block_size - len(encoded_text)
            encoded_text += [self.ctoi['<pad>']] * padding_length

        x = torch.tensor(encoded_text[:-1], dtype=torch.long)
        y = torch.tensor(encoded_text[1:], dtype=torch.long)

        return x, y


# Function to load checkpoint
def load_checkpoint(model, optimizer, ckpt_path):
    if os.path.isfile(ckpt_path):
        print(f"Loading checkpoint '{ckpt_path}'")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")
        return start_epoch
    else:
        print(f"No checkpoint found at '{ckpt_path}', starting from scratch.")
        return 0


# Function to estimate loss
@torch.no_grad()
def estimate_loss(model, data_loader):
    model.eval()
    losses = []
    for x, y in data_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# Training loop
def train_model(model, train_loader, val_loader, trainer_config):
    model.train()
    optimizer = model.configure_optimizers(trainer_config)
    scaler = torch.cuda.amp.GradScaler()
    accumulation_steps = 4  # Adjust as needed

    start_epoch = 0
    if trainer_config.ckpt_path:
        start_epoch = load_checkpoint(model, optimizer, trainer_config.ckpt_path)

    total_steps = len(train_loader)
    for epoch in range(start_epoch, trainer_config.max_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{trainer_config.max_epochs}")
        for i, (x, y) in enumerate(progress_bar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                logits, loss = model(x, y)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})

            if (i + 1) % trainer_config.eval_interval == 0 or (i + 1) == total_steps:
                val_loss = estimate_loss(model, val_loader)
                print(f"\nEpoch {epoch+1}, Step {i + 1}/{total_steps}, Training Loss: {loss.item() * accumulation_steps:.4f}, Validation Loss: {val_loss:.4f}")
                model.train()

        # Save model at end of epoch
        if trainer_config.ckpt_path:
            os.makedirs(os.path.dirname(trainer_config.ckpt_path), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, trainer_config.ckpt_path)
            print(f"Saved model checkpoint to '{trainer_config.ckpt_path}'")


# Function to generate predictions and save
def generate_predictions(model, ctoi, itoc, input_json_file, output_json_file, device):
    os.makedirs(output_folder, exist_ok=True)

    # input_json_file = 'preprocessed/splits/val.json'
    # output_json_file = os.path.join(output_folder, f"{os.path.splitext(json_file)[0]}_energy_result.json")

    # Read input JSON file
    with open(input_json_file, 'r', encoding='utf-8') as f:
        input_data_list = json.load(f)

    # Prepare list for output
    output_data_list = []

    # Generate predictions
    for data_item in input_data_list:
        encoding_input = data_item['encoding']
        true_energy = data_item.get('energy')  # Real energy
        predicted_energy = generate_target(model, encoding_input, ctoi, itoc, device, max_new_tokens=50)  # Model prediction

        output_data_list.append({
            'encoding': encoding_input,
            'true_energy': true_energy,
            'predicted_energy': predicted_energy
        })

    # Save to output JSON file
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data_list, f, ensure_ascii=False, indent=4)

    print(f"Saved predictions to '{output_json_file}'")


# Main execution
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Setting for CrystalGPT-1', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-mode', type=str, default='train', help='train or generate')
    parser.add_argument('-n_layer', type=int, default=12)
    parser.add_argument('-n_head', type=int, default=12)
    parser.add_argument('-n_embd', type=int, default=768)
    parser.add_argument('-block_size', type=int, default=256)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-max_epochs', type=int, default=5)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-learning_rate', type=float, default=1e-4)
    parser.add_argument('-weight_decay', type=float, default=0.1)
    parser.add_argument('-eval_interval', type=int, default=1000)

    parser.add_argument('-random_seed', type=int, default=42)
    parser.add_argument('-gpu_index', type=int, default=0)

    parser.add_argument('-train_data_file', type=str, default='preprocessed/splits/train.json')
    parser.add_argument('-val_data_file', type=str, default='preprocessed/splits/val.json')
    parser.add_argument('-test_data_file', type=str, default='preprocessed/splits/test.json')
    parser.add_argument('-ckpt_path', type=str, default='preprocessed/checkpoints/final_model.pt')
    parser.add_argument('-generate_file', type=str, default='preprocessed/predicted_energy_result.json')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # Set device and specify GPU index
    device = torch.device(f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Read training data
    print("Loading training data...")
    with open(args.train_data_file, 'r', encoding='utf-8') as f:
        train_data_list = json.load(f)
    
    # Read validation data
    print("Loading validation data...")
    with open(args.val_data_file, 'r', encoding='utf-8') as f:
        val_data_list = json.load(f)
    
    print(f"Number of training samples: {len(train_data_list)}")
    print(f"Number of validation samples: {len(val_data_list)}")
    
    # Build vocabulary
    print("Building vocabulary...")
    chars = set()
    
    for data in train_data_list:
        encoding = data['encoding']
        target = data['energy']  # or 'area'
        combined_text = f"{encoding}\n{target}\n"
        chars.update(combined_text)
    
    # Convert set to sorted list
    chars = sorted(list(chars))
    ctoi = {c: i for i, c in enumerate(chars)}
    itoc = {i: c for i, c in enumerate(chars)}
    
    # Add special tokens
    ctoi['<pad>'] = len(ctoi)
    itoc[len(itoc)] = '<pad>'
    ctoi['<unk>'] = len(ctoi)
    itoc[len(itoc)] = '<unk>'
    vocab_size = len(ctoi)
    print(f'Vocabulary size: {vocab_size}')

    gpt_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        embd_pdrop=args.dropout,
        resid_pdrop=args.dropout,
        attn_pdrop=args.dropout
    )

    # Create model instance
    model = GPT(gpt_config).to(device) 

    if args.mode == 'train':
        trainer_config = TrainerConfig(
            max_epochs=args.max_epochs,  
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            eval_interval=args.eval_interval, 
            ckpt_path=args.ckpt_path  # Path to save model checkpoints
        )

        # Prepare datasets and dataloaders
        train_dataset = LargeTextDataset(train_data_list, ctoi, block_size)
        val_dataset = LargeTextDataset(val_data_list, ctoi, block_size)
        
        train_loader = DataLoader(train_dataset, batch_size=trainer_config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=trainer_config.batch_size, num_workers=4, pin_memory=True)

        # Start training
        train_model(model, train_loader, val_loader, trainer_config)
    
        # Save final model
        os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
        torch.save(model.state_dict(), args.ckpt_path)
        print(f"Saved final model to '{args.ckpt_path}'")
    
    elif args.mode == 'generate':
        # Generate predictions
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        generate_predictions(model, ctoi, itoc, args.test_data_file, args.generate_file, device)

    else:
        raise NotImplementedError
