



# SpatialAlign: Aligning Dynamic Spatial Relationships in Video Generation

<p align="center">
  <!-- 可放 teaser 图 / pipeline 图 -->
  <img src="assets/teaser.png" width="95%">
</p>

<p align="center">
  <!-- 徽章：论文/项目页/许可证/Stars 等 -->
  <a href="<paper>"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg"></a>
  <a href="https://fengming001ntu.github.io/SpatialAlign/"><img src="https://img.shields.io/badge/Project-Page-blue.svg"></a>
</p>

---

## 📌 Overview
Most text-to-video (T2V) generators prioritize aesthetic quality, but often ignoring the spatial constraints in the generated videos. In this work, we present SPATIALALIGN, a self-improvement framework that enhances T2V models’ capabilities to depict Dynamic Spatial Relationships (DSR) specified in text prompts. We present a zeroth-order regularized Direct Preference Optimization (DPO) to fine-tune T2V models towards better alignment with DSR. Specifically, we design DSR-SCORE, a geometry-based metric that quantitatively measures the alignment between generated videos and the specified DSRs in prompts, which is a step forward from prior works that rely on VLM for evaluation. We also conduct a dataset of text-video pairs with diverse DSRs to facilitate the study. Extensive experiments demonstrate that our fine-tuned model significantly outperforms the baseline in spatial relationships. 

---

## ⚙️ Setup

### Install Environment via Anaconda (Recommended)
```bash
conda create -n spatialalign python=3.10
conda activate spatialalign
pip install -r requirements.txt
```

## Training
📥 **Training data (tensors + metadata)**: [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/fengming001_e_ntu_edu_sg/IgCz3bCNT5WLRpCaqvvaIiLBAbNXp3lpfXNBjROqfp4UM3k?e=6dz89e)
