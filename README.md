# Medical Named Entity Recognition (NER) with BioBERT 

## Project Overview
The purpose of this project is to automatically identify and classify medical entities from clinical and biomedical text using Natural Language Processing. The model detects three types of entities — Pathogens, Medicines, and Medical Conditions — from raw text input.

The project covers the complete pipeline from data preprocessing and model fine-tuning to a fully deployed interactive web application. It was implemented using Python along with HuggingFace Transformers, PyTorch, and the pre-trained BioBERT model, deployed as a web app using Dash.

## Dataset Information
The dataset used is Corona2.json — a custom annotated biomedical corpus containing real-world medical text with character-level entity span annotations.

| Field | Description |
|-------|-------------|
| content | Raw biomedical text |
| annotations | Character-level entity spans with start, end, and tag |
| tag_name | Entity type label |

### Entity Distribution in Original Dataset:
| Entity Type | CountMedical |
|-------------|--------------|
| Condition | 134| 
| Medicine | 94 | 
| Pathogen | 67 |

Since the original dataset contained only 31 real-world examples, template-based data augmentation was applied to generate 200 additional synthetic examples — bringing the total to 231 examples.

## Project Approach
This project follows a complete NLP pipeline for token-level sequence labeling using the **BIO tagging scheme**.

1) The project starts with **Data Cleaning** where annotations are checked for noisy short labels, irrelevant strings, and whitespace boundary misalignments in character spans
2) **Template-based Data Augmentation** was then applied using curated medical vocabularies of 30 Pathogens, 28 Medicines, and 28 Medical Conditions, combined with 15 sentence templates to generate realistic synthetic training examples
3) After augmentation, **BIO Tagging and Label Alignment** was performed. Since BioBERT uses WordPiece subword tokenization, a two-step alignment process was used — first mapping character-level annotations to word-level BIO labels, then aligning word-level labels to subword tokens using ***word_ids()***
4) The dataset was split into training and testing sets using an **85:15 ratio**, and features were extracted using ***DataCollatorForTokenClassification*** for dynamic padding
5) The fine-tuned model was then deployed as a fully interactive **Dash web application** supporting text input, file upload, entity highlighting, and grouped entity summary cards

## Why This Pipeline Was Chosen
The pipeline was carefully designed based on the characteristics of biomedical text and the limitations of the small dataset.

1) **BioBERT** was selected because it is pre-trained on PubMed abstracts and PMC full-text articles, giving it a strong domain-specific understanding of medical terminology that general BERT models lack
2) **BIO Tagging** was used because it is the standard scheme for Named Entity Recognition and cleanly represents multi-token entity spans
3) **Subword Label Alignment** was implemented to correctly handle WordPiece tokenization — only the first subword of each word receives the real label, while continuation subwords are masked with ***-100*** to be ignored during loss computation
4) **Template-based Augmentation** was chosen because it preserves exact annotation boundaries and produces grammatically valid biomedical sentences
5) **Dash** was chosen for deployment because it allows building interactive Python-native web applications without requiring a separate frontend framework

### This pipeline is highly suitable for this project because:
The dataset is small and domain-specific — BioBERT compensates with pre-trained biomedical knowledge. Precise label alignment prevents training signal corruption. Augmentation increases diversity without introducing annotation noise. Dash enables rapid deployment directly from Python.

## Machine Learning Workflow
The complete workflow starts with dataset loading and cleaning. After augmenting and preparing annotations, BIO labels are assigned and aligned with BioBERT's tokenizer. The model is then fine-tuned, evaluated, and deployed as an interactive web application.
### The project pipeline follows this flow:
*Data Loading → Data Cleaning → Data Augmentation → BIO Label Assignment → Subword Alignment → Train-Test Split → BioBERT Fine-Tuning → Hyperparameter Optimization → Model Evaluation → Model Export → Dash Deployment*

## Label Schema
The project uses the BIO (Beginning-Inside-Outside) tagging scheme with 3 entity types resulting in 7 total labels.
| Label | ID |
|-------|----|
| O | 0 |
| B-Medicine | 1 |
| I-Medicine | 2 |
| B-Pathogen | 3 |
| I-Pathogen | 4 |
| B-MedicalCondition | 5 |
| I-MedicalCondition | 6 |

## Training Configuration
| Parameter | Value |
|-----------|-------|
| Base Model | dmis-lab/biobert-base-cased-v1.1 |
| Learning Rate | 3e-5 |
| Batch Size | 4 |
| Epochs | 10 |
| Max Sequence Length | 512 |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.1 |
| LR Scheduler | Cosine |
| Best Model Metric | Eval Loss |

## Model Evaluation
Different evaluation metrics were used to measure model performance at the entity span level using ***seqeval***.

1) Precision measures what fraction of predicted entities are correct
2) Recall measures what fraction of true entities were successfully detected
3) F1-Score is the harmonic mean of Precision and Recall
4) A per-entity Classification Report was generated to evaluate performance separately for Medicine, Pathogen, and MedicalCondition

****Token-level accuracy was intentionally avoided as it overestimates performance due to the dominance of ***O*** tokens.****

## Deployment — Dash Web Application
The fine-tuned BioBERT model was deployed as an interactive web application built with Dash and Dash Bootstrap Components.

App Features:
* Paste any biomedical text directly into the input area
* Upload .txt or .pdf files for automatic text extraction
* Click Analyse Text to run the NER model in real time
* View color-highlighted entities inline within the original text
*  View grouped entity summary cards showing all detected Pathogens, Medicines, and Medical Conditions

### Entity Color Coding:
| Entity | Color |
|--------|-------|
| 🦠 Pathogen | Red |
| 💊 Medicine | Blue |
| 🏥 Medical Condition | Green |

### To run the app locally:
*pip install dash dash-bootstrap-components transformers torch PyPDF2
python app.py*
Then open your browser and go to ***http://localhost:8050***

## Technologies Used
* Python
* PyTorch
* HuggingFace Transformers
* HuggingFace Datasets
* BioBERT (dmis-lab/biobert-base-cased-v1.1)
* seqeval
* NumPy
* Dash
* Dash Bootstrap Components
* PyPDF2
* Kaggle (Training Environment)

## Real-World Applications
This project demonstrates how NLP and transformer models can automate information extraction from biomedical text. Similar systems are actively used in:

* Clinical decision support systems
* Drug discovery and pharmacovigilance
* Epidemic and outbreak monitoring
* Electronic Health Record (EHR) processing
* Biomedical literature mining

## Future Improvements
In the future, this project can be improved by:

1) Fine-tuning on larger annotated biomedical corpora such as NCBI Disease or BC5CDR
2) Adding more entity types such as Genes, Anatomy, and Dosage
3) Deploying the application to a cloud platform such as Heroku or Hugging Face Spaces
4) Adding export functionality to download detected entities as a CSV or JSON file
5) Applying advanced cross-validation techniques for more robust evaluation

## Conclusion
The Medical NER project successfully demonstrates the complete pipeline for biomedical Named Entity Recognition using a fine-tuned BioBERT model, from raw annotated data to a fully deployed interactive web application. By combining precise BIO label alignment, template-based data augmentation, domain-specific pretraining, and a Dash-powered frontend, the system reliably identifies Pathogens, Medicines, and Medical Conditions from any biomedical text input.
The project highlights key NLP and deployment concepts including subword tokenization, sequence labeling, BIO tagging, transformer fine-tuning, entity-level evaluation, and web application development. It serves as a strong practical implementation of end-to-end biomedical NLP for real-world clinical and research applications.
