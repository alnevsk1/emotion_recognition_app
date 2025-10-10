import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from datasets import load_dataset
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

class DushaDataset(Dataset):
    def __init__(self, hf_dataset, window_size_sec=0.5, sample_rate=16000, n_mels=128, hop_length=160, n_fft=1024, device='cpu'):
        self.dataset = hf_dataset
        self.window_size = int(sample_rate * window_size_sec)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.label2idx = {'angry':0, 'sad':1, 'neutral':2, 'positive':3, 'other':4}
        self.device = device
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        entry = self.dataset[idx]
        waveform = torch.tensor(entry['audio']['array']).float()
        waveform = waveform.to(self.device)
        label = self.label2idx[entry['emotion']]
        if waveform.shape[-1] < self.window_size:
            waveform = nn.functional.pad(waveform, (0, self.window_size - waveform.shape[-1]))
        else:
            waveform = waveform[:self.window_size]
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
        )(waveform)
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        return mel_spec, label

class CRNNWithAttention(nn.Module):
    def __init__(self, n_mels=128, n_classes=5, hidden_dim=128, num_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
        )
        self.rnn = nn.GRU((n_mels//4)*64, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim*2, 1)
        self.fc = nn.Linear(hidden_dim*2, n_classes)
    
    def forward(self, x):
        h = self.cnn(x)
        h = h.permute(0, 3, 1, 2)
        batch_size, time_steps, channels, n_mels = h.size()
        h = h.reshape(batch_size, time_steps, channels*n_mels)
        rnn_out, _ = self.rnn(h)
        attn_weights = torch.softmax(self.attn(rnn_out).squeeze(-1), dim=-1)
        emb = torch.sum(rnn_out * attn_weights.unsqueeze(-1), dim=1)
        out = self.fc(emb)
        return out

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'model_epoch{epoch}_loss{loss:.4f}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Saved checkpoint: {path}")

def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'Loaded model from {checkpoint_path} (epoch {epoch}, loss {loss:.6f})')
    return epoch, loss


def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for feats, label in tqdm(dataloader, desc='Eval', leave=False):
            feats = feats.unsqueeze(1).to(device)
            label = label.to(device)
            logits = model(feats)
            preds = torch.argmax(logits, dim=-1)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print("Accuracy: {:.4f}, F1 (weighted): {:.4f}".format(acc, f1))
    print("Classification report:")
    print(classification_report(y_true, y_pred))
    return acc, f1


def train(model, optimizer, criterion, train_loader, device, start_epoch, num_epochs, checkpoint_dir, val_loader=None):
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for feats, label in tqdm_loader:
            feats = feats.unsqueeze(1).to(device)
            label = label.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            tqdm_loader.set_postfix({'Batch Loss': loss.item()})
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        if val_loader:
            print("Validation metrics:")
            acc, f1 = evaluate(model, val_loader, device)
        save_checkpoint(model, optimizer, epoch+1, avg_loss, checkpoint_dir)


def main(checkpoint_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    ds = load_dataset("xbgoose/dusha", split="train[:90%]")
    ds_val = load_dataset("xbgoose/dusha", split="train[90%:]")
    train_set = DushaDataset(ds, window_size_sec=0.5, sample_rate=16000, n_mels=64, hop_length=160, n_fft=400, device=device)
    val_set = DushaDataset(ds_val, window_size_sec=0.5, sample_rate=16000, n_mels=64, hop_length=160, n_fft=400, device=device)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4)

    model = CRNNWithAttention(n_mels=64, n_classes=5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=4e-5)
    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = "./checkpoints"

    start_epoch = 0
    if checkpoint_path:
            start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path, device)
    num_epochs = 20
    train(model, optimizer, criterion, train_loader, device, start_epoch, num_epochs, checkpoint_dir, val_loader)


if __name__ == "__main__":
    # To continue from checkpoint, pass the file path as argument.
    # Example: main("./checkpoints/model_epoch10_loss0.8631")
    main("./checkpoints/model_epoch11_loss0.7723.pt")