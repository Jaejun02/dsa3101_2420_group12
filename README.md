# ESG Data Extraction and Performance Analysis

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Installation Guide](#installation-guide)
  - [Linux Environment](#linux-environment)
  - [Windows/macOS/Other Environment](#windowsmacosother-environment)
  - [Docker Setup](#docker-setup)
- [Usage Instructions](#usage-instructions)
  - [Demo Application](#demo-application)
  - [Interface and Configuration](#interface-and-configuration)
- [Dashboard Integration](#dashboard-integration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview
This project automates the extraction and performance analysis of **Environmental, Social, and Governance (ESG)** data from unstructured ESG reports. Leveraging advanced **Large Language Models (LLMs)** and **Natural Language Processing (NLP)** techniques, the solution extracts key ESG metrics to provide fast, consistent, and scalable performance insights for informed decision-making across various industries.

## Key Features
- **Web Scraping:**  
  Automates the retrieval of ESG sustainability reports from corporate websites such as [responsibilityreports.com](https://www.responsibilityreports.com/).

- **Data Extraction:**  
  Utilizes utility functions and predefined scoring mechanisms to extract ESG information. Integrates ESG scores from LSEG and LinkedIn-sourced data for benchmarking and validation.

- **Data Pre-processing:**  
  Employs Optical Character Recognition (OCR) and layout analysis with the pre-trained **MinerU** model to extract text from unstructured PDF reports. Also leverages LLMs for image and chart analysis.

- **Prompt Engineering:**  
  Optimizes LLM prompts to derive meaningful insights by fine-tuning input formatting, enhancing model response accuracy.

- **Semantic Search:**  
  Implements a **FAISS-based** semantic search that supports both keyword-based and context-driven retrieval of ESG documents.

- **Visualization:**  
  Provides interactive Power BI dashboards for a structured display of ESG performance metrics.

- **User Interface:**  
  Features a Gradio-based web interface for straightforward interaction with search and extraction functionalities.

## Installation Guide

### Linux Environment
**Note:** For non-Linux systems, please refer to the Docker or WSL instructions below.
1. **Create a virtual environment** (using `conda` or `venv`).
2. **Install Magic-PDF:**
   ```bash
   pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com
3. **Install Hugging Face Hub:**
   ```bash
   pip install huggingface_hub
   ```
4. **Download required models:**
   ```bash
   python download_models_hf.py
   ```
5. **Configure GPU mode:**
   - Navigate to your home directory (e.g., `/home/username`) and open `magic-pdf.json`.
   - Modify the value of `"device-mode"`:
     ```json
     {
       "device-mode": "cuda"
     }
     ```
6. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
7. **Configure Environment Variables:**
  Set up your `HF_TOKEN` in the `.env` file:
  ```bash
  HF_TOKEN=your_hf_token_goes_here
  ```
8. **Run the Application**
  ```bash
  python run.py
  ```

### Windows/MacOS/Other Environment
**Note:** The **vllm** component is currently Linux-only. For other OS:
* **Docker Container**:
  Run the project within a Linux Container
* **WSL on Windows**
  Install and configure Windows Subsystem for Linux (WSL) to execute the project in a Linux environment.

### Docker Setup
1. **Prepare the Data Directory Structure:**
  ```bash
  /data/  
  ├── pdfs/      # Place ESG PDF Reports here
  ├── config/    # Include your config.json file here
  └── results/   # Processed results will be stored here
  ```
2. **Build the Docker Container:**
   ```bash
   docker build -t app .
   ```
3. **Run the Docker Container:**
   ```bash
   docker run --gpus all --env-file .env -v ./data:/data app
   ```
4. **Access the Extraction Results:**
  Retrieve the extraction results in `/data/results/`.

## Demo Project Usage Instructions

### Starting the Demo App
You can start the demo application using either Docker or direct execution.
**Using Docker:**
```bash
docker build -t demo-app -f demo_dockerfile/Dockerfile .  # Build the container
docker run --gpus all -p 7860:7860 demo-app  # Run the container
```

**Direct Execution:**
```bash
python demo.py
```

Then, visit [http://localhost:7860](http://localhost:7860) to open the Gradio-based User Interface.

### Interface Configurations
* **LLM Model Selection:**
- Choose the primary Large Language Model (LLM) for processing.
- Optionally enable GPU quantization to balance speed, memory usage, and accuracy.

* **Semantic Search Model Selection**
- Select the Semantic Search Model (Currently,single semantic search model is supported.)

* **LLM Parameters Configurations**
  * **Temperature**: Control output randomness (e.g., `0.2` for deterministic responses, `0.8` for diverse outputs.)
  * **Top-p (Nucleus Sampling):** Limits token selection probability (e.g., `0.9 includes the top 90% of probable tokens.)
  * **Max Tokens:** Specifies the maximum tokens for responses.

* **Search Parameters**
  * **Top-k:** Number of top search results to consider.
  * **Rerank-k:** Number of resu;ts re-evaluated for relevance.
  * **Alpha**: Balances keyword-based and semantic search contributions to importance.

* **Model Initialization:**
Click **Initiliaze Models** and wait for confirmation that the models have loaded successfully. (The successfulness of the 3 Key components will be displayed.)

* **File Uploads:**
Upload one or more PDF/text files to start the extraction process.

* **Data Extraction:**
Click **Extract ESG Data** to generate:
  * ESG Extraction Results
  * Sentiment Analysis Results
  * ESG Scoring Results

* **Data Download**
Processed CSV files can be downloaded for further analysis.

## Dashboard Integration
Follow the detailed instructions in `instructions.txt` (located in the database folder [here](https://github.com/Jaejun02/dsa3101_2420_group12/tree/main/database)) to set up and run the interactive Power BI dashboards. Visit the [Power BI documentation](https://www.microsoft.com/en-us/power-platform/products/power-bi/) for further guidance.

## License
This project is licensed under the MIT License.

## Acknowledgements
This project was developed with support from various open-source tools, models, and data providers:

- [responsibilityreports.com](https://www.responsibilityreports.com/) – ESG report sourcing.
- [MinerU](https://github.com/opendatalab/MinerU) – OCR and layout analysis.
- [Hugging Face](https://huggingface.co/) – Pre-trained LLMs and model APIs.
- [FAISS](https://github.com/facebookresearch/faiss) – Semantic search via vector indexing.
- [Gradio](https://gradio.app/) – Rapid deployment of ML demos.
- [Power BI](https://powerbi.microsoft.com/) – Dashboarding and data visualization.