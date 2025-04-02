# Automating ESG Data Extraction and Performance Analysis

This project aims to automate the extraction and performance analysis of **Environmental, Social, and Governance (ESG)** data from unstructured ESG reports using **Large Language Models (LLMs)** and **Natural Language Processing (NLP)** techniques. The model extracts key metrics, enabling more efficient **ESG performance analysis**. This approach significantly enhances decision-making by providing scalable, automated solutions for analyzing complex ESG data across industries.

### Features

### Web Scraping
- Automates the retrieval of ESG sustainability reports from corporate websites such as [reponsibility.com](https://www.responsibilityreports.com/) for analysis.
### Data Extraction
- Extracts ESG-related information from ESG reports using utility functions and predefined scoring mechanisms.
- Incorporates ESG scores from LSEG and LinkedIn-scraped data for benchmarking and validation.
### Data Pre-processing
- Utilizes Optical Character Recognition and Layout Analysis using pre-trained model MinerU to extract textual data from unstructured ESG PDF reports
- Leveraged LLMs to analyse images, charts and visual data for textual content
### Prompt Engineering
- Optimizes LLM prompts to extract meaningful ESG insights.
- Fine-tunes input formatting for improved model responses.
### Search Engine
- Implements **FAISS-based** semantic search for ESG document retrieval.
- Supports both **keyword-based and vector-based** searches.
### Power BI Dashboarding
- Visualizes ESG performance metrics using interactive Power BI dashboards.
- Provides structured insights for stakeholders.
### Gradio Interface
- Deploys a user-friendly web interface for testing NLP models.
- Allows easy interaction with search and extraction functionalities.

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
