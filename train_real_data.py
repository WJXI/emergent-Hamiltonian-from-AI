import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import time
from mini_af3_model import MiniAF3ScoreModel, sample_spherical_noise_and_target

class SpinGlassDataset(Dataset):
    def __init__(self, data_dir="SO3_TrainSet_40k"):
        files = sorted(glob.glob(f"{data_dir}/chunk_*.npz"))
        self.sequences, self.spins = [], []
        for f in files:
            data = np.load(f, allow_pickle=True)
            self.sequences.extend(data['sequences'])
            self.spins.extend(data['spins'])
        print(f"✅ loading：{len(self.sequences)} sequences。")

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        seq_tensor = torch.tensor([0 if c == 'A' else 1 for c in self.sequences[idx]], dtype=torch.long)
        snap_idx = np.random.randint(len(self.spins[idx]))
        return seq_tensor, torch.tensor(self.spins[idx][snap_idx], dtype=torch.float32)

def collate_fn(batch):
    max_len = 50
    B = len(batch)
    padded_seqs = torch.zeros((B, max_len), dtype=torch.long)
    padded_spins = torch.zeros((B, max_len, 3), dtype=torch.float32)
    masks = torch.zeros((B, max_len), dtype=torch.bool)
    for i, (seq, spins) in enumerate(batch):
        N = len(seq)
        padded_seqs[i, :N] = seq; padded_spins[i, :N] = spins; masks[i, :N] = True
    return padded_seqs, padded_spins, masks

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 start training | device: {device}")
    
    dataset = SpinGlassDataset("SO3_TrainSet_40k")
    
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)
    
    # test
    model = MiniAF3ScoreModel(c_s=64, c_z=32, num_blocks=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 20
    print("\n⏳ start training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (seq, S_0, mask) in enumerate(dataloader):
            seq, S_0, mask = seq.to(device), S_0.to(device), mask.to(device)
            B, N = seq.shape
            
            optimizer.zero_grad()
            
            # sampling time t ~ U(1e-4, 1.0)
            t = torch.rand(B, 1).to(device) * 0.9999 + 0.0001
            t_expanded = t.unsqueeze(2).expand(B, N, 1)
            
            # SO3 Target
            S_t, target_score = sample_spherical_noise_and_target(S_0, t_expanded)
            
            # SO3 Score
            pred_score = model(seq, S_t, t, mask)
            
            # Likelihood Weighting ,cancel 1/t sigularity

            loss_all = t_expanded * (pred_score - target_score)**2
            
            loss = loss_all[mask.unsqueeze(-1).expand_as(loss_all)].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | Weighted Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1}/{epochs} | average Loss: {avg_loss:.4f} | time: {time.time()-start_time:.1f} s")
        torch.save(model.state_dict(), f"mini_af3_epoch_{epoch+1}.pt")
        
    print("🎉 sucessful train")

