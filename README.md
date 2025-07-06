# SkinTect

**SkinTect** is a real-time allergen detection system for cosmetic products using OCR, deep learning, and natural language processing. It extracts ingredient information from product labels, identifies allergens using BioBERT, and provides personalized product recommendations through clustering algorithms.

---

## Technologies Used

| **Module**          | **Technology**                |
|-----------------|---------------------------|
| Text Detection  | YOLOv5 (CNN)              |
| Text Recognition| CRNN (CNN + RNN + CTC)    |
| NLP Analysis    | BioBERT (NER & Parsing)   |
| Frontend        | HTML5, CSS3               |
| Backend         | Python (Flask / FastAPI)  |

---

## Project Structure
```
SkinTect/
├── app.py # Main application server
├── requirements.txt # Python dependencies
├── best.pt # Trained YOLOv5 model
├── allergen_detector.py # Main logic for detection and analysis
├── allergen_dictionary.json # List of known allergens
├── reverse_synonym_index.json # NLP synonym mapping
├── final_biobert_model/ # BioBERT model & tokenizer
│ ├── config.json
│ ├── model.safetensors
│ └── ...
├── html/ # Web interface
│ ├── Homepage.html
│ ├── scanner.html
│ └── ...
├── yolov5/ # YOLOv5 local code
└── .gitignore
```
Some files can't be upladed due to file size limitation. To compensate for this, here are links to the said files:
NLP Fine-tuned Model: https://drive.google.com/drive/folders/1mYosogc0y3JRjavnRWnjQTf_q11XQVnV?usp=sharing

---

## Getting Started

### 1. Clone the repository
git clone https://github.com/your-username/SkinTect.git
cd SkinTect

### 2. Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

### 3. Install dependencies
py -3.9 -m pip install -r requirements.txt

### 4. Run the app
py -3.9 app.py


