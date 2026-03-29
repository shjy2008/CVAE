# Conditional Variational Autoencoder (CVAE) for Sign Language MNIST

**Tech Stack:** Python, TensorFlow, Keras  
**Data Source:** https://www.kaggle.com/datasets/datamunge/sign-language-mnist

This project implements a **Conditional Variational Autoencoder (CVAE)** trained on the **Sign Language MNIST dataset** to perform three generation tasks:

1. Generate images of **consonants vs. vowels**  
2. Generate images of a **specific letter**  
3. Generate images with **rotation angles** 0°, 90°, 180°, 270°  

Three separate scripts and models correspond to each task:  

- `myModel_task1.py` / `myModel_task1.keras`  
- `myModel_task2.py` / `myModel_task2.keras`  
- `myModel_task3.py` / `myModel_task3.keras`  

---

## Model Architecture

### Encoder
- Input: 28×28 grayscale image + conditional information  
  - Shape: `(28, 28, 1 + add_dimension)`  
  - `add_dimension` = number of conditional channels (1 for binary, num_classes for multi-class)  
- Three convolutional layers with filters `[128, 64, 128]`, kernel size 3, stride 2, ReLU activation  
- Flatten → Dense layer producing latent vector `(mean + log variance)`  
- Latent dimension: 512  
- Batch normalization after each conv layer  

### Decoder
- Input: latent vector + conditional information `(latent_dim + add_dimension)`  
- Dense layer → reshape → two transposed conv layers `[128, 64]`  
- Final transposed conv layer (1 filter) → reconstructed image  
- ReLU activation in hidden layers  

### Loss Function
- **Total Loss = Reconstruction Loss + KL Divergence**  
- Reconstruction: binary cross-entropy  
- KL divergence ensures latent distribution is close to standard normal  

---

## Conditional Information by Task

- **Task 1:** Consonant (0) or vowel (1)  
  - Binary labels concatenated with latent vector  
- **Task 2:** Specific letter (0–23, one-hot encoding)  
  - One-hot vector concatenated with latent vector  
- **Task 3:** Rotation angle (0°, 90°, 180°, 270°; one-hot)  
  - Rotated images generated using `tf.image.rot90`  
  - One-hot vector concatenated with latent vector  

---

## Summary of Architecture

| Model | Task1 | Task2 | Task3 |
|-------|-------|-------|-------|
| Classes | 2 | 24 | 4 |
| Conditional Info | 0/1 | One-hot 24 labels | One-hot 4 labels |
| Latent Dimension | 512 | 512 | 512 |
| Encoder | 3 conv layers (128,64,128) + 1 dense | Same | Same |
| Decoder | 1 dense + 3 transposed conv (128,64,1) | Same | Same |
| Loss | Reconstruction + KL divergence | Same | Same |
| Learning Rate | 0.001 | 0.001 | 0.001 |

---

## Performance Overview

- Trained 100 epochs per task (Task 3 dataset 4× larger due to rotations)  

### Task 1
- Generates consonant/vowel images  
- Good examples resemble letters, bad examples are blurry or mixed  
<img width="480" height="720" alt="image" src="https://github.com/user-attachments/assets/69eddc72-b495-4bec-a1b7-1f66d6b14eac" />

### Task 2
- Generates specific letters  
- Most images are clear; letters L, P, Q are more difficult  
- Clear patterns for simple letters (e.g., C), complex patterns harder (e.g., Q)  
<img width="480" height="720" alt="image" src="https://github.com/user-attachments/assets/dda2c02e-d907-4572-be9f-d8fca27f6e64" />

### Task 3
- Generates rotated images  
- Blurry due to unknown letter identity  
- Rotation angle is distinguishable  
- Performance could improve with more layers/filters  
<img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/ac183989-0e68-45db-af34-4a6b2fd1ab78" />

---

## CVAE vs. VAE

### Advantages of CVAE
- Conditional generation based on labels  
- Produces clearer outputs for labeled datasets  

### Use Cases for CVAE
- Generate images of specific animals, human faces by age/gender, or any labeled dataset  

### Limitations
- Requires labeled data  
- Not suitable for fully unsupervised tasks  

### Use Cases for VAE
- Unlabeled data or anomaly detection  
- Example: network traffic anomaly detection  

---

## Conclusion

- CVAE enables generation of **consonant/vowel**, **specific letters**, and **rotated images**  
- Task 2 (specific letters) achieves the clearest outputs due to guided conditional information  
- The project highlights the strengths of CVAE for **conditional generation** and demonstrates scenarios where standard VAE is more appropriate  
