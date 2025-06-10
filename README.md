# SkinTect

**Real-Time Allergen Detection in Cosmetic Products Using OCR, Deep Learning, and NLP**

SkinTect is a real-time allergen detection system that empowers users—especially those with allergies or sensitive skin—to make informed decisions about cosmetic products. Using a combination of OCR, NLP, and clustering algorithms, the system analyzes ingredient labels from cosmetic products to identify potential allergens, harmful chemical interactions, and provide personalized product recommendations.

---

## Project Overview

**SkinTect** addresses the limitations of barcode scanners and static allergen databases by offering:

- Real-time ingredient label analysis from images
- Allergen detection using BioBERT (NLP)
- Chemical interaction detection for harmful ingredient combinations
- Personalized recommendations based on user allergen profiles
- Fast processing with a target response time of under 2 seconds

---

## Technologies Used

- **YOLOv5** – Text detection from cosmetic labels
- **CRNN** – Text recognition for OCR
- **BioBERT** – Named Entity Recognition for allergen detection
- **K-Means + Hierarchical Clustering** – Personalized recommendations
- **Python**, **OpenCV**, **PyTorch**, **Transformers**, **Scikit-learn**

---

## Repository Structure

```plaintext
SkinTect/
│
├── README.md
├── .gitignore
├── docs/                    # Reports and documentation
│   └── sprint-1/
│   └── sprint-2/
│
├── models/                  # Pretrained or fine-tuned models
│   ├── yolo/
│   ├── crnn/
│   └── biobert/
│
├── modules/                 # Functional modules
│   ├── ocr/
│   ├── allergen_detection/
│   ├── clustering/
│   └── chemical_interaction/
│
├── ui/                      # User interface files
├── scripts/                 # Evaluation, testing scripts
└── requirements.txt         # Project dependencies
```
## Sprint Planning
Sprint tasks, assignees, durations, and story points are documented in our Sprint Planning Sheet and are aligned with out Gantt chart and project milestones.

## Team Members
* Maria Consuelo Abalos Jr.
* Princess Lucille Acnam
* Byvien Dy
* Ma. Janna Malangen
