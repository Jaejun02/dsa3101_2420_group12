## Table of Contents

- [About the Project](#about-the-project)
- [Overview](#overview)
- [Installation Guide](#installation-guide)
  - [Linux Environment](#linux-environment)
  - [Windows Environment](#windows-environment)
  - [Running Docker Container](#running-docker-container)
- [Demo Project Usage Instructions](#demo-project-usage-instructions)
- [Technologies Used](#technologies-used)
- [Models and Methods Used](#models-and-methods-used)
- [Acknowledgements](#acknowledgements)

# About the Project
### Automating ESG Data Extraction and Performance Analysis

This project aims to automate the extraction and performance analysis of **Environmental, Social, and Governance (ESG)** data from unstructured ESG reports using **Large Language Models (LLMs)** and **Natural Language Processing (NLP)** techniques. The model extracts key metrics, enabling more efficient **ESG performance analysis**. This approach significantly enhances decision-making by providing scalable, automated solutions for analyzing complex ESG data across industries.

## Overview

| Feature                | Description |
|------------------------|-------------|
| **Web Scraping** | - Automates the retrieval of ESG sustainability reports from corporate websites such as [reponsibilityreports.com](https://www.responsibilityreports.com/) for analysis |
| **Data Extraction**  | - Extracts ESG-related information from ESG reports using utility functions and predefined scoring mechanisms <br /> - Incorporates ESG scores from LSEG and LinkedIn-scraped data for benchmarking and validation |
| **Data Pre-processing** | - Utilizes Optical Character Recognition (OCR) and Layout Analysis using pre-trained model **MinerU** to extract textual data from unstructured ESG PDF reports <br /> - Leveraged LLMs to analyse images, charts and visual data for textual content.|
| **Prompt Engineering**  | - Optimizes LLM prompts to extract meaningful ESG insights <br /> - Fine-tunes input formatting for improved model responses.|
| **Search Engine**     | - Implements **FAISS-based** semantic search for ESG document retrieval <br /> - Supports both **keyword-based and semantic** searches. |
| **Power BI Dashboarding** | - Visualizes ESG performance metrics using interactive Power BI dashboards <br /> - Provides structured insights for stakeholders |
| **Gradio Interface**  | - Deploys a user-friendly web interface for testing NLP models <br /> - Allows easy interaction with search and extraction functionalities |

## Installation Guide
### Linux Environment
1. Create a virtual environment (`conda` or `venv`)
2. Install magic-pdf
   ```bash
   pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com
   ```
3. Install Hugging Face Hub
   ```bash
   pip install huggingface_hub
   ```
4. Download required models
   ```bash
   python download_models_hf.py
   ```
5. Configure GPU mode
   - Navigate to user directory `\home\username`, open `magic-pdf.json`
   - Modify the value of `"device-mode"`:
    ```json
    {
      "device-mode": "cuda"
    }
    ```
6. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
8. Launch the demo
   ```bash
   python demo.py
   ```

### Windows Environment
1. Follow the [MinerU setup for Windows](https://github.com/opendatalab/MinerU/blob/master/docs/README_Windows_CUDA_Acceleration_en_US.md).
2. Repeat steps 6-7 from Linux setup above.

### Running Docker Container
1. Run the following commands
```bash
docker build -t demo-app . # Build the container
docker run --gpus all -p 7860:7860 demo-app # Run the container
```
2. Go to [http://localhost:7860](http://localhost:7860)

## Demo Project Usage Instructions
Clear examples of how to use the project once set up/installed.
## Technologies Used
A list of tools, libraries, and frameworks utilized.
## Models and Methods Used
Details on any algorithms, models, or methods used
## Acknowledgements
Any external resources
