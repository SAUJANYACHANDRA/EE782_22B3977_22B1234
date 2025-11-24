# EE782_22B3977_22B0978

Sketch-to-Photo Translation using CycleGAN
Paper-Style Implementation (128×128, 6-ResNet Blocks, LSGAN)

This project implements a full CycleGAN model for converting hand-drawn face sketches → realistic face photos and photos → sketches, following the architecture described in the CycleGAN research paper.

📘 Overview

Unpaired image-to-image translation (no paired sketches/photos needed)

Generators: 3-layer encoder → 6 ResNet blocks → 3-layer decoder

Discriminators: 5-layer PatchGAN

Losses: LSGAN (MSE), Cycle-Consistency (L1), Identity loss (L1)

Image size: 128×128 (as used in the paper)

Fixed sample evaluation: same sketches saved and used every epoch

Checkpointing: models and example outputs saved every 10 epochs

📂 Dataset
Sketch dataset

CUHK Face Sketch Database (CUFS)
Contains 606 aligned sketches of human faces.

Photo dataset

Human Faces Dataset (Kaggle)
Large diverse face photos used as the target domain.

Folder structure must be:

sketches/
    img1.jpg
    img2.jpg
    ...

faces/
    photo1.jpg
    photo2.jpg
    ...

    raining

Run:

```
python Train.py
```

The script automatically:

Loads both datasets

Caches fixed sketches for consistent evaluation

Saves sample outputs to outputs_full/

Saves checkpoints to checkpoints_full/

Begins training for 500 epochs (configurable)

Initial (epoch 0), intermediate, and final results are saved automatically.

🧪 Generating Samples

After training, the final model cyclegan_full_final.pth is saved.
You can run the sample generation function in the script or load the model:

```
ckpt = torch.load("cyclegan_full_final.pth")
G = GeneratorFull(6).to(device)
G.load_state_dict(ckpt["G_sketch2photo"])
G.eval()
```

🛠 Model Summary
Generator (Sketch ↔ Photo)

ReflectionPad2d

Conv → BN → ReLU

Conv → BN → ReLU

Conv → BN → ReLU

6 ResNet blocks (Residual learning)

ConvTranspose → BN → ReLU

ConvTranspose → BN → ReLU

Tanh output

Discriminator

5-layer PatchGAN

LeakyReLU activations

Last layer outputs a patch map (not a single scalar)

📷 Output Examples

Training automatically stores images at:

outputs_full/epoch_010.png
outputs_full/epoch_020.png
...


Each saved image shows input sketches and their generated photos.

📦 Checkpoints

Checkpoints saved in:
```
checkpoints_full/checkpoint_epoch_010.pth
checkpoints_full/checkpoint_epoch_020.pth
...
```

Final trained model:

cyclegan_full_final.pth

📜 Citations

CycleGAN Paper

@article{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A.},
  year={2017},
  journal={ICCV}
}


CUF Sketch Dataset

@dataset{cufs,
  title={CUHK Face Sketch Database},
  year={2011}
}
