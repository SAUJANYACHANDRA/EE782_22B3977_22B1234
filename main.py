"""
CycleGAN FULL - Paper Implementation Version (with Fixed Sample Evaluation)
- Aligns architecture with the paper: 128x128 images, 6 Resnet blocks, 5-layer Discriminator.
- Uses LSGAN (MSE) and L1 Cycle Consistency Loss.
- Caches a fixed batch of input sketches (NUM_SAMPLES=6) for consistent visual comparison across epochs.
"""

import os
import glob
import warnings
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

# Suppress PIL warnings
warnings.filterwarnings('ignore', message='Palette images with Transparency')

# ============================================================================
# CONFIGURATION - BASED ON PAPER
# ============================================================================
SKETCH_PATH = "./sketches"  # CUHK Face Sketch Database path
PHOTO_PATH = "./faces"     # Human Faces dataset path

# Training settings
BATCH_SIZE = 6            # Reduced for better GPU memory management at 128x128
IMAGE_SIZE = 128          # Paper specifies 128x128
NUM_EPOCHS = 180          # Paper notes convergence around 500 epochs
SAVE_EVERY = 10           # Save/evaluate every 10 epochs
NUM_SAMPLES = 6           # Number of fixed samples to use for evaluation

# Loss Weights (Standard CycleGAN hyper-parameters)
LAMBDA_CYCLE = 10.0       
LAMBDA_IDENTITY = 5.0     

# Output directories
OUTPUT_DIR = "./outputs_full"
CHECKPOINT_DIR = "./checkpoints_full"
SAMPLE_CACHE_PATH = "./fixed_sample_sketch.pth" # File to save the fixed sample batch

# ============================================================================
# Setup & Utilities
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("="*70)
print("CYCLEGAN FULL - FIXED SAMPLE EVALUATION")
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Target Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}, Fixed Samples: {NUM_SAMPLES}")
print("="*70)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def validate_paths():
    """Check if dataset paths exist and contain images"""
    errors = []
    if not os.path.exists(SKETCH_PATH) or not os.path.exists(PHOTO_PATH):
        errors.append(f"❌ Check paths: SKETCH_PATH={SKETCH_PATH}, PHOTO_PATH={PHOTO_PATH}")
    
    # Simple check for image files
    sketch_files = glob.glob(SKETCH_PATH + "/**/*.jpg", recursive=True) + glob.glob(SKETCH_PATH + "/**/*.png", recursive=True)
    photo_files = glob.glob(PHOTO_PATH + "/**/*.jpg", recursive=True) + glob.glob(PHOTO_PATH + "/**/*.png", recursive=True)
    
    if len(sketch_files) == 0: errors.append(f"❌ No images found in sketch path: {SKETCH_PATH}")
    if len(photo_files) == 0: errors.append(f"❌ No images found in photo path: {PHOTO_PATH}")
    
    if errors:
        print("\n".join(errors))
        return False
    
    print(f"✓ Found {len(sketch_files)} sketch images and {len(photo_files)} photo images")
    return True

# ============================================================================
# Dataset (Using 128x128 Resize)
# ============================================================================
class ImageFolder(Dataset):
    def __init__(self, root, img_size=IMAGE_SIZE):
        self.files = glob.glob(root + "/**/*.jpg", recursive=True) + \
                     glob.glob(root + "/**/*.png", recursive=True) + \
                     glob.glob(root + "/**/*.jpeg", recursive=True) + \
                     glob.glob(root + "/**/*.JPG", recursive=True)
        
        if len(self.files) == 0:
            raise ValueError(f"No images found in {root}")
        
        # NOTE: Resizing to 128x128 as specified in the paper
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert("RGB")
            return self.transform(img)
        except Exception as e:
            # Fallback for corrupted images
            print(f"Error loading {self.files[idx]}: {e}. Retrying next image.")
            return self.__getitem__((idx + 1) % len(self.files))

# ============================================================================
# Model Components - Accurate to Paper
# ============================================================================

# ResnetBlock (Implementing F(x)=y-x, the residual function)
class ResnetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Paper specifies: normalization, two convolutional layers, and a ReLU layer in between
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # Reflection padding is common practice for CycleGAN
            nn.Conv2d(channels, channels, 3, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, padding=0),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x) # Residual connection

# Generator (3 conv encoder, 6 Resnet blocks, 3 deconv decoder)
class GeneratorFull(nn.Module):
    def __init__(self, num_residual_blocks=6): 
        super().__init__()
        
        # 1. Encoder (3 Convolutional Layers)
        encoder = [
            # Initial conv layer (C7S1-32)
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 32, kernel_size=7, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Downsampling 1 (D64)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Downsampling 2 (D128)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        ]
        
        # 2. Resnet Blocks (R128 * 6)
        resnet_blocks = [ResnetBlock(128) for _ in range(num_residual_blocks)]
        
        # 3. Decoder (3 Transpose Convolutional Layers)
        decoder = [
            # Upsampling 1 (U64)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Upsampling 2 (U32)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Output layer (C7S1-3)
            nn.ReflectionPad2d(3),
            nn.Conv2d(32, 3, kernel_size=7, padding=0),
            nn.Tanh() # Tanh activation on the final output
        ]
        
        self.model = nn.Sequential(*encoder, *resnet_blocks, *decoder)
    
    def forward(self, x):
        return self.model(x)

# Discriminator (5 convolutional layers)
class DiscriminatorFull(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            # 1. Input layer (C64) - No BatchNorm on the first layer
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            
            # 2. Layer (C128)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            
            # 3. Layer (C256)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            
            # 4. Layer (C512)
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            
            # 5. Output/Classification Layer (C1) - Maps to a single output patch
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

# ============================================================================
# Initialization and Loss Functions
# ============================================================================
G_sketch2photo = GeneratorFull(num_residual_blocks=6).to(device)
G_photo2sketch = GeneratorFull(num_residual_blocks=6).to(device)
D_photo = DiscriminatorFull().to(device)
D_sketch = DiscriminatorFull().to(device)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

G_sketch2photo.apply(init_weights)
G_photo2sketch.apply(init_weights)
D_photo.apply(init_weights)
D_sketch.apply(init_weights)

# Loss functions
criterion_GAN = nn.MSELoss() # LSGAN for stability
criterion_cycle = nn.L1Loss() # L1 Loss for Cycle Consistency
criterion_identity = nn.L1Loss()

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_sketch2photo.parameters(), G_photo2sketch.parameters()),
    lr=0.0002, betas=(0.5, 0.999) # Matching paper's hyper-parameters
)
optimizer_D_photo = torch.optim.Adam(D_photo.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_sketch = torch.optim.Adam(D_sketch.parameters(), lr=0.0002, betas=(0.5, 0.999))

print("✓ Models initialized")

# ============================================================================
# Visualization (FIXED SAMPLES)
# ============================================================================

def denormalize(img):
    img = img.cpu().detach().numpy()
    img = (img * 0.5 + 0.5).transpose(1, 2, 0)
    return img.clip(0, 1)

def cache_fixed_samples(sketch_loader, num_samples=NUM_SAMPLES):
    """Samples a fixed batch of images and saves it to disk for consistent evaluation."""
    print(f"   Caching {num_samples} fixed samples for consistent evaluation...")
    try:
        # Get the fixed samples from the first batch
        fixed_batch = next(iter(sketch_loader))[:num_samples]
        torch.save(fixed_batch, SAMPLE_CACHE_PATH)
        print(f"   ✓ Fixed sample batch saved to: {SAMPLE_CACHE_PATH}")
    except StopIteration:
        print("   ⚠️ Cannot cache samples, sketch loader is empty.")
    except Exception as e:
        print(f"   ❌ Error caching samples: {e}")

def generate_samples(generator, epoch=0, num_samples=NUM_SAMPLES):
    """Loads the fixed samples and generates translations for comparison."""
    if not os.path.exists(SAMPLE_CACHE_PATH):
        print("   ⚠️ Cannot generate samples, fixed sample cache not found.")
        return

    generator.eval()
    
    # Load the fixed batch (always the same set of images)
    sketch_batch = torch.load(SAMPLE_CACHE_PATH).to(device)
    
    plt.figure(figsize=(10, 2.5 * num_samples))
    with torch.no_grad():
        generated = generator(sketch_batch)
    
    for i in range(num_samples):
        # Input Sketch
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(denormalize(sketch_batch[i]))
        plt.title(f"Input Sketch {i+1}")
        plt.axis('off')
        
        # Generated Photo
        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(denormalize(generated[i]))
        plt.title(f"Generated Photo {i+1}")
        plt.axis('off')
    
    plt.suptitle(f"Results at Epoch {epoch}", fontsize=14)
    plt.tight_layout()
    
    filename = f"{OUTPUT_DIR}/epoch_{epoch:03d}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   📸 Saved: {filename}")
    generator.train()

def save_checkpoint(epoch):
    """Save model checkpoint"""
    checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch:03d}.pth"
    torch.save({
        'epoch': epoch,
        'G_sketch2photo': G_sketch2photo.state_dict(),
        'G_photo2sketch': G_photo2sketch.state_dict(),
        'D_photo': D_photo.state_dict(),
        'D_sketch': D_sketch.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D_photo': optimizer_D_photo.state_dict(),
        'optimizer_D_sketch': optimizer_D_sketch.state_dict(),
    }, checkpoint_path)
    print(f"   💾 Checkpoint: {checkpoint_path}")

# ============================================================================
# Training Function
# ============================================================================
def train_cyclegan_full(sketch_loader, photo_loader, epochs=NUM_EPOCHS, 
                       lambda_cycle=LAMBDA_CYCLE, lambda_identity=LAMBDA_IDENTITY):
    
    print(f"\n🚀 Starting training for {epochs} epochs")
    print(f"   Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}, Batch Size: {sketch_loader.batch_size}")
    print("-" * 70)
    
    for epoch in range(epochs):
        epoch_losses = {'G': 0, 'D_photo': 0, 'D_sketch': 0, 'cycle': 0, 'batches': 0}
        min_len = min(len(sketch_loader), len(photo_loader))

        for i, (sketch_batch, photo_batch) in enumerate(zip(sketch_loader, photo_loader)):
            # Ensure batches are the same size
            sketch = sketch_batch.to(device)
            photo = photo_batch.to(device)
            batch_size = min(sketch.size(0), photo.size(0))
            sketch = sketch[:batch_size]
            photo = photo[:batch_size]
            
            # --- Setup Labels (LSGAN) ---
            real_label = torch.ones(D_photo(photo).size()).to(device)
            fake_label = torch.zeros(D_photo(photo).size()).to(device)
            
            # ==================================
            # Train Generators 
            # ==================================
            optimizer_G.zero_grad()
            
            # 1. Identity Loss: G_A(A) -> A, G_B(B) -> B (Weighted)
            loss_identity = (criterion_identity(G_sketch2photo(photo), photo) + 
                             criterion_identity(G_photo2sketch(sketch), sketch)) * lambda_identity
            
            # 2. GAN Loss: G wants D to think fake images are real (Target = 1)
            fake_photo = G_sketch2photo(sketch)
            loss_GAN_sketch2photo = criterion_GAN(D_photo(fake_photo), real_label)
            
            fake_sketch = G_photo2sketch(photo)
            loss_GAN_photo2sketch = criterion_GAN(D_sketch(fake_sketch), real_label)
            
            # 3. Cycle Consistency Loss: S -> P -> S and P -> S -> P (Weighted)
            loss_cycle = (criterion_cycle(G_photo2sketch(fake_photo), sketch) + 
                          criterion_cycle(G_sketch2photo(fake_sketch), photo)) * lambda_cycle
            
            loss_G = loss_GAN_sketch2photo + loss_GAN_photo2sketch + loss_cycle + loss_identity
            
            loss_G.backward()
            optimizer_G.step()
            
            # ==================================
            # Train Discriminators
            # ==================================
            
            # D_photo
            optimizer_D_photo.zero_grad()
            loss_D_real_photo = criterion_GAN(D_photo(photo), real_label)
            loss_D_fake_photo = criterion_GAN(D_photo(fake_photo.detach()), fake_label)
            loss_D_photo = 0.5 * (loss_D_real_photo + loss_D_fake_photo)
            loss_D_photo.backward()
            optimizer_D_photo.step()
            
            # D_sketch
            optimizer_D_sketch.zero_grad()
            loss_D_real_sketch = criterion_GAN(D_sketch(sketch), real_label)
            loss_D_fake_sketch = criterion_GAN(D_sketch(fake_sketch.detach()), fake_label)
            loss_D_sketch = 0.5 * (loss_D_real_sketch + loss_D_fake_sketch)
            loss_D_sketch.backward()
            optimizer_D_sketch.step()
            
            # Update epoch metrics
            epoch_losses['G'] += loss_G.item()
            epoch_losses['D_photo'] += loss_D_photo.item()
            epoch_losses['D_sketch'] += loss_D_sketch.item()
            epoch_losses['cycle'] += loss_cycle.item() / lambda_cycle
            epoch_losses['batches'] += 1
            
            if (i + 1) % 50 == 0:
                 print(f"  > Batch [{i+1:4d}/{min_len}] | G: {loss_G.item():.3f} | Dp: {loss_D_photo.item():.3f} | Ds: {loss_D_sketch.item():.3f}")

        # Average losses
        num_batches = epoch_losses['batches']
        for key in ['G', 'D_photo', 'D_sketch', 'cycle']:
            epoch_losses[key] /= num_batches
        
        # Print epoch summary
        print(f"Epoch [{epoch+1:3d}/{epochs}] | "
              f"G_Total: {epoch_losses['G']:.3f} | "
              f"D_Photo: {epoch_losses['D_photo']:.3f} | "
              f"D_Sketch: {epoch_losses['D_sketch']:.3f} | "
              f"Cycle_L1: {epoch_losses['cycle']:.3f}")
        
        # Save samples and checkpoints (Every 10 epochs or at the start)
        if (epoch + 1) % SAVE_EVERY == 0 or epoch == 0:
            print(f"   📸 Saving outputs...")
            generate_samples(G_sketch2photo, epoch + 1, num_samples=NUM_SAMPLES)
            save_checkpoint(epoch + 1)
            
    print("\n" + "="*70)
    print("✅ Training complete!")
    print("="*70)
    
    return G_sketch2photo

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    if not validate_paths():
        exit(1)
    
    try:
        # Initialize loaders
        num_workers = 0 if os.name == 'nt' else 2
        sketch_loader = DataLoader(ImageFolder(SKETCH_PATH, img_size=IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"))
        photo_loader = DataLoader(ImageFolder(PHOTO_PATH, img_size=IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"))
        
        # Cache fixed samples for consistent evaluation
        cache_fixed_samples(sketch_loader, num_samples=NUM_SAMPLES)

        # Generate initial samples (Epoch 0)
        print("\n📸 Generating initial samples (untrained)...")
        generate_samples(G_sketch2photo, epoch=0, num_samples=NUM_SAMPLES)
        
        trained_model = train_cyclegan_full(sketch_loader, photo_loader)
        
        # Save final model
        final_path = "cyclegan_full_final.pth"
        torch.save({'G_sketch2photo': G_sketch2photo.state_dict(), 'G_photo2sketch': G_photo2sketch.state_dict(),}, final_path)
        print(f"\n💾 Final model saved: {final_path}")
        
        # Generate final samples
        print("\n📸 Generating final samples...")
        generate_samples(trained_model, epoch=NUM_EPOCHS, num_samples=NUM_SAMPLES)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user. Saving emergency checkpoint...")
        save_checkpoint(9999)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
