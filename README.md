#  Player Psychological State Analysis from Facial Expressions

##  Introduction
This project proposes a novel system for inferring players’ psychological states from facial expression sequences in real-world sports videos using a time-series deep learning framework integrated with face image quality assessment (FIQA).

---

##  Overview

Monitoring psychological states in sports is challenging due to:

- Motion blur, occlusion, and unstable camera conditions
- Rapid emotional changes during gameplay  
- Lack of continuous temporal modeling in traditional methods  

 This project addresses these issues by:

- Modeling psychological state as a **continuous latent process**  
- Integrating face image quality as a **reliability signal**  
- Learning temporal dynamics using **LSTM**  

---

##  System Pipeline
<img width="1525" height="404" alt="pipeline" src="https://github.com/user-attachments/assets/5e4cfb64-c2e4-4d4d-b7f2-e9224dd2fc6c" />
<!-- <p align="center">
  <img src="demo/pipeline.png" width="600"/>
</p> -->

The system consists of the following stages:

1. **Face Detection**  
   - Detect faces in each frame using RetinaFace 


2. **Face Image Quality Assessment (FIQA)**  
   - Assign a quality score to each detected face  
   - Used as a *soft confidence weight*, not a hard filter  

3. **Feature Extraction**  
   - Extract facial expression embeddings using improved ResNet50  

4. **Time-Series Modeling**  
   - Model temporal evolution using LSTM  
   - Learn a latent psychological state vector  

5. **Latent State Analysis**  
   - Compute interpretable metrics:  
     - **LSI** (Latent Stability Index)  
     - **TDI** (Temporal Drift Index)  
     - **LVI** (Latent Variability Index)  

---

##  Dataset

The system is trained using:

- **FER2013** (pretraining)  
- **Custom Sports Dataset**  
  - Labels: *fatigue, stress, excitement*  
  - Collected from FIFA & UEFA videos  
- **Real Match Videos**  
  - Used for time-series modeling  

---

##  Demo

 [Watch on YouTube](https://youtu.be/hov_bqoBBvM)
---

##  Requirements

```bash
tensorflow
opencv-python
retinaface
