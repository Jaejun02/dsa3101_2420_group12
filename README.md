## Table of Contents
- [About the Project](#about-our-project)
- [Overview](#overview)
- [Installation Guide](#installation-guide)
  - [Linux Environment](#linux-environment)
  - [Windows Environment](#windows-environment)
  - [Running Docker Container](#running-docker-container)
- [Demo Project Usage Instructions](#demo-project-usage-instructions)
- [Acknowledgements](#acknowledgements)

# About Our Project
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
| **Data Pre-processing** | - Utilizes Optical Character Recognition and Layout Analysis using pre-trained model MinerU to extract <br /> textual data from unstructured ESG PDF reports <br /> - Leveraged LLMs to analyse images, charts and visual data for textual content.|
| **Prompt Engineering**  | - Optimizes LLM prompts to extract meaningful ESG insights <br /> - Fine-tunes input formatting for improved model responses.|
| **Search Engine**     | - Implements **FAISS-based** semantic search for ESG document retrieval <br /> - Supports both **keyword-based and semantic** searches. |
| **Power BI Dashboarding** &nbsp; &nbsp; | - Visualizes ESG performance metrics using interactive Power BI dashboards <br /> - Provides structured insights for stakeholders |
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

# Demo Project Usage Instructions

## Accessing the Interface
- Connect to `localhost:7860`. This will automatically direct you to the Gradio-based user interface.

## Selecting the LLM Model
- Choose the primary Large Language Model (LLM) for processing.
- There is an option to apply quantization to the GPU, which reduces memory consumption and speeds up inference. However, this comes at the expense of slight performance and accuracy trade-offs.

## Choosing the Semantic Search Model
- Select the Semantic Search Model. Currently, only one option is available.

## Configuring the LLM Parameters
- **Temperature**: Controls the randomness of the model’s output.
  - Lower value (e.g., `0.2`) → More deterministic responses.
  - Higher value (e.g., `0.8`) → Increased diversity.
- **Top-p (Nucleus Sampling)**: Determines the probability threshold for selecting tokens.
  - A value of `0.9` means the model considers only the top 90% of probable next words, reducing randomness but maintaining variety.
- **Max Tokens**: Specifies the maximum number of tokens (words or subwords) the model is allowed to generate in a single response.

## Setting the Search Parameters
- **Top-k**: Defines how many top search results to consider.
  - A higher value broadens the search space but may introduce less relevant results.
- **Rerank-k**: Specifies how many of the top-k search results should be re-evaluated and reordered for improved relevance.
- **Alpha**: A weighting factor that balances between keyword-based search and semantic similarity-based retrieval.

## Initializing Models
1. Click on **"Initialize Models"**.
2. Wait for the system to complete the initialization process.
3. The interface will confirm successful initialization of three key components.

## Uploading Files for Processing
- Upload the required **PDF** or **text** files.
- The system supports multiple file uploads.

## Extracting ESG Data
- Click on **"Extract ESG Data"** to initiate the extraction process.
- The system will generate and display the following results:
  - **ESG Extraction Results**
  - **Sentiment Analysis Results**
  - **ESG Scoring Results**

## Downloading Processed Data
- Download the generated **CSV** files as needed for further analysis and reporting.

## Acknowledgements
This project was developed with the support of various open-source tools, models, and data providers:

- [responsibilityreports.com](https://www.responsibilityreports.com/) – For ESG report sourcing
- [MinerU](https://github.com/opendatalab/MinerU) – OCR and layout analysis for ESG reports
- [Hugging Face](https://huggingface.co/) – For hosting pre-trained LLMs and model APIs
- [FAISS](https://github.com/facebookresearch/faiss) – Vector-based semantic search
- [Gradio](https://gradio.app/) – For providing a simple interface to deploy ML demos
- [Power BI](https://powerbi.microsoft.com/) – For enabling dashboarding and data visualization


  

