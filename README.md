# Pretty Pictures, Shallow Parsing: The Impact of Syntactic Structure on Diffusion Models

This repository contains the code for the paper *“Pretty Pictures, Shallow Parsing: The Impact of Syntactic Structure on Diffusion Models”*.

The study evaluates how a text-to-image diffusion model handles prompt with manipulations on:

- Word Order  
- Long-Distance Dependencies (LDD)  
- Syntactic Ambiguity  

---

## Requirements

- Python 3.13  
- NVIDIA GPU recommended (original experiments used an A100 80GB VRAM)

For lower-memory hardware, models can be adapted using quantization.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

(On Windows: `.venv\Scripts\activate`)

Install dependencies:

```bash
pip install -r requirements.txt
```

Download required models:

```bash
python -m spacy download en_core_web_sm
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Set your Hugging Face token in `data_generation/image_generation.py`.

---

## Repository Structure

```text
.
├── prompts/
│   ├── wo.txt
│   ├── amb.txt
│   ├── ldd.txt
│   └── ldd_prompts.py
│
├── data_generation/
│   ├── questions.py
│   ├── image_generation.py
│   └── VLM_question_answering.py
│
├── analysis/
│   ├── ambiguity_analysis.py
│   ├── ldd_analysis.py
│   ├── evaluation.py
│   └── word_order_analysis.ipynb
│
├── requirements.txt
└── README.md
```

---

## Reproducing the Study

Run the following steps in order:

```bash
python data_generation/questions.py
python data_generation/image_generation.py
python data_generation/VLM_question_answering.py
```

---

## Statistical Analysis

```bash
python analysis/evaluation.py
python analysis/ambiguity_analysis.py
python analysis/ldd_analysis.py
jupyter notebook analysis/word_order_analysis.ipynb
```

---

## Citation

If you use this code, please cite:

```bibtex
@misc{Ceka2026prettypictures,
  title={Pretty Pictures, Shallow Parsing: The Impact of Syntactic Structure on Diffusion Models},
  author={Ceka, Chiara and Gaddo, Tiziano and La Marra, Marco and Zamparelli, Roberto},
  journal={},
  year={2026},
  volume={},
  pages={}
}
```
