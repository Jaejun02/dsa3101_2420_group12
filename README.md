# Automating ESG Data Extraction and Performance Analysis

This project aims to automate the extraction and performance analysis of **Environmental, Social, and Governance (ESG)** data from unstructured ESG reports using **Large Language Models (LLMs)** and **Natural Language Processing (NLP)** techniques. The model extracts key metrics, enabling more efficient **ESG performance analysis**. This approach significantly enhances decision-making by providing scalable, automated solutions for analyzing complex ESG data across industries.

### Features:
- **Web Scraping**: 
- **Data Extraction**: 
- **Data Pre-processing**:
- **Prompt Engineering**: 
- **Search Engine**: 
- **PowerBi Dashboarding**:
- **Gradio Interface**:

### Installation Steps (Linux Environment)
1. Create a virtual environment (conda or venv)
2. Run `pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com`
3. Run `pip install huggingface_hub`
4. Run `python download_models_hf.py`
5. Navigate to user directory `\home\username`, open `magic-pdf.json` and modify the value of `"device-mode"`
```
{
  "device-mode": "cuda"
}
```
6. Run `pip install -r requirements.txt`
7. Run `python demo.py`

### Installation (Windows Environment)
1. Follow steps on the [mineru repo](https://github.com/opendatalab/MinerU/blob/master/docs/README_Windows_CUDA_Acceleration_en_US.md).
2. Same as steps 6-7 above.

### Demo Project Usage Instructions
Clear examples of how to use the project once set up/installed.
### Technologies Used
A list of tools, libraries, and frameworks utilized.
### Models and Methods Used
Details on any algorithms, models, or methods used
### Acknowledgements
Any external resources
