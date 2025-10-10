import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm.notebook import tqdm
import os
import matplotlib.pyplot as plt

class DushaDataset(Dataset):
    def __init__(self, dataset, target_sr, max_len_sec, n_mels, n_fft, hop_length, label_map):
        self.dataset = dataset
        self.target_sr = target_sr
        self.max_len_sec = max_len_sec
        self.label_map = label_map
        
        # Define Mel Spectrogram transform - This part is correct.
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        # The resampler is removed from __init__

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # 1. Load and Get Original Sampling Rate
        waveform = sample['audio']['array']
        original_sr = sample['audio']['sampling_rate'] # Get SR from the individual sample
        
        # Convert to tensor
        waveform = torch.tensor(waveform).float().unsqueeze(0)
        
        # 2. Resample if necessary (Moved from __init__)
        if original_sr != self.target_sr:
            # Create the resampler on-the-fly for the specific sample
            resampler = T.Resample(original_sr, self.target_sr)
            waveform = resampler(waveform)

        # 3. Pre-processing (Padding/Trimming)
        max_samples = int(self.target_sr * self.max_len_sec)
        
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples] # Truncate
        elif waveform.shape[1] < max_samples:
            padding = max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding)) # Pad

        # 4. Feature Extraction (Mel Spectrogram)
        mel_spec = self.mel_spectrogram(waveform).squeeze(0)
        log_mel_spec = T.AmplitudeToDB()(mel_spec)

        # Add a channel dimension for the CNN: (1, N_MELS, N_FRAMES)
        features = log_mel_spec.unsqueeze(0)
        
        # Get label
        label_id = self.label_map[sample['emotion']]

        return features, torch.tensor(label_id, dtype=torch.long)
    
class SER_Model(nn.Module):
    def __init__(self, num_classes, n_mels):
        super(SER_Model, self).__init__()
        
        # 1. Convolutional Block (Feature Extractor from Mel Spectrogram)
        # Input shape: (Batch, 1, n_mels, n_frames)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2)), # Output: (B, 32, n_mels/2, n_frames/2)
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2)), # Output: (B, 64, n_mels/4, n_frames/4)
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2)) # Output: (B, 128, n_mels/8, n_frames/8)
        )
        
        # Calculate the size of the feature vector after CNN
        # Feature height after 3 MaxPool2d layers (2*2): n_mels // 8
        cnn_output_height = n_mels // 8
        rnn_input_size = 128 * cnn_output_height # 128 channels * height

        # 2. Recurrent Block (Temporal Modeling)
        # We process the output of the CNN as a sequence of frames
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True, # Bi-LSTM captures context in both directions
            dropout=0.3
        )

        # 3. Classifier Block
        # Bi-LSTM output size: 2 * hidden_size (due to bidirectional)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes) # Final layer for 5 emotions
        )

    def forward(self, x):
        # x shape: (B, 1, N_MELS, N_FRAMES)
        
        # CNN forward pass
        x = self.cnn(x)
        # x shape: (B, Channels, Height, Frames) e.g., (B, 128, 16, 32)
        
        B, C, H, W = x.shape
        
        # Reshape for RNN: combine Channels and Height into feature dimension
        # New shape: (B, N_FRAMES, Features) -> (B, W, C*H)
        x = x.permute(0, 3, 1, 2).contiguous() # (B, W, C, H)
        x = x.view(B, W, C * H)
        
        # RNN forward pass
        # The output contains hidden states for each time step
        rnn_output, _ = self.rnn(x) # rnn_output shape: (B, N_FRAMES, 2*128)
        
        # Global Pooling: Average the sequence of hidden states across all frames
        # This provides an utterance-level representation
        avg_pool = torch.mean(rnn_output, dim=1) # (B, 2*128)
        
        # Classifier forward pass
        logits = self.classifier(avg_pool)
        
        return logits

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for features, labels in tqdm(dataloader, desc="Training"):
        features, labels = features.to(device), labels.to(device)

        # Forward pass
        logits = model(features)
        loss = criterion(logits, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        
        # Calculate accuracy
        _, predicted = torch.max(logits, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Evaluating"):
            features, labels = features.to(device), labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item() * features.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def visualize_temporal_emotion(model, audio_path, label_map, device, max_len_sec, target_sr):
    # Load and Preprocess a single audio file
    waveform, sr = torchaudio.load(audio_path)
    
    # Standard resampling
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    # Convert to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Padding/Trimming (same as in Dataset)
    max_samples = int(target_sr * max_len_sec)
    original_length_sec = waveform.shape[1] / target_sr
    
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    
    # Feature Extraction
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=target_sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel_spec = T.AmplitudeToDB()(mel_spectrogram(waveform)).unsqueeze(0)
    features = log_mel_spec.to(device)
    
    model.eval()
    with torch.no_grad():
        # --- Modified Forward Pass for Temporal Output ---
        x = model.cnn(features) # (1, Channels, Height, Frames)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, W, C * H) # (1, N_FRAMES_RNN, Features)
        
        # Pass through RNN
        rnn_output, _ = model.rnn(x) # (1, N_FRAMES_RNN, 2*128)
        
        # Apply the final classifier to *each time step* in the RNN output sequence
        # We need to reshape the output to apply the classifier: (1*N_FRAMES_RNN, 2*128)
        temporal_logits = model.classifier(rnn_output.view(-1, rnn_output.size(-1)))
        
        # Convert logits to probabilities (Softmax)
        temporal_probs = torch.softmax(temporal_logits, dim=1)
        
        # Reshape back to (N_FRAMES_RNN, NUM_CLASSES)
        temporal_probs = temporal_probs.view(W, NUM_CLASSES).cpu().numpy()

    # --- Visualization ---
    
    # Calculate time axis (since CNN/Pooling reduced the frame count, time steps are fewer)
    time_points = np.linspace(0, min(original_length_sec, max_len_sec), temporal_probs.shape[0])
    
    plt.figure(figsize=(12, 6))
    
    # Plot probability of each emotion over time
    for i, label in enumerate(label_map):
        plt.plot(time_points, temporal_probs[:, i], label=label)
        
    plt.title(f"Temporal Emotion Probability for Audio (Duration: {original_length_sec:.2f}s)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # --- Configuration ---
    # Check for GPU and set device
    print("start")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Dataset parameters
    SAMPLING_RATE = 16000           # Target sampling rate
    MAX_AUDIO_LENGTH = 4.0          # Max audio length in seconds
    NUM_CLASSES = 5
    EMOTION_LABELS = ['neutral', 'angry', 'positive', 'sad', 'other']

    # Feature extraction parameters (Mel Spectrogram)
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 512

    # Training parameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    
    print("Loading dataset...")
    raw_datasets = load_dataset("xbgoose/dusha", split={"train": "train", "test": "test"})

    # Map string labels to integers
    label_to_id = {label: i for i, label in enumerate(EMOTION_LABELS)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    # --- Class Balance and Weighted Loss Calculation (from provided data) ---
    train_counts = {
        'neutral': 0.527,
        'angry': 0.209,
        'positive': 0.161,
        'sad': 0.090,
        'other': 0.013
    }

    # Calculate inverse frequency weights
    total_samples_train = 150000
    class_counts = {k: int(v * total_samples_train) for k, v in train_counts.items()}
    class_counts_list = [class_counts[label] for label in EMOTION_LABELS]

    # Calculate inverse of counts
    inv_counts = 1.0 / np.array(class_counts_list)

    # Normalize weights (e.g., by the smallest weight, or just use the raw inverse)
    # Using raw inverse for better scaling in loss function
    weights = inv_counts / np.min(inv_counts)

    CLASS_WEIGHTS = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print("\nCalculated Class Weights (for CrossEntropyLoss):")
    for i, (label, weight) in enumerate(zip(EMOTION_LABELS, CLASS_WEIGHTS.cpu().numpy())):
        print(f"  {i} ({label}): {weight:.2f}")
    
    # Create Dataset and DataLoader instances
    train_dataset = DushaDataset(raw_datasets["train"], SAMPLING_RATE, MAX_AUDIO_LENGTH, N_MELS, N_FFT, HOP_LENGTH, label_to_id)
    test_dataset = DushaDataset(raw_datasets["test"], SAMPLING_RATE, MAX_AUDIO_LENGTH, N_MELS, N_FFT, HOP_LENGTH, label_to_id)

    # Set num_workers=0 for debugging
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    example_features, _ = train_dataset[0]
    print(f"Example feature shape (C, H, W): {example_features.shape}")
    
    # Instantiate model, criterion, optimizer
    model = SER_Model(NUM_CLASSES, N_MELS).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Main Training Loop ---
    print("\nStarting Training...")
    best_test_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Save best model based on test loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'best_ser_model.pth')
            print("  --> Saved best model checkpoint.")

    print("\nTraining complete.")

if __name__ == '__main__':
    main()