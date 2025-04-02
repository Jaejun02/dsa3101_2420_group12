from extract import extract, sentiment_analysis
from preprocessing.extract_pdf import extract_text_from_pdf
from postprocessing.postprocess import postprocess
import pandas as pd
import os
import tempfile
from typing import List, Tuple, Dict, Any, Optional
import torch
import json
import numpy as np
from transformers import AutoTokenizer
import re
from search_engine.engine import ESGSearchEngine
import nltk
from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable is not set.")
login(token=hf_token)

def get_available_gpus():
    """
    Detect available GPUs and return options for dropdown.
    
    Returns:
        List of integers for available GPU counts
    """
    try:
        gpu_count = torch.cuda.device_count()
        # Return options from 1 to gpu_count
        return list(range(1, gpu_count + 1)) if gpu_count > 0 else [0]
    except:
        # Return [0] if no CUDA available
        return [0]

def get_params():
    """
    Reads in predefined configurations to run the script.

    Args:
        None
    
    Returns:
        main_model_name: Name of the main model to initialize
        search_model_name: Name of the search model to initialize
        gpu_count: Number of GPUs to use
        params: Dictionary of parameters
    """
    with open('./data/config/config.json') as file:
        config = json.load(file)
    return config['main_model_name'], config['search_model_name'], config['gpu_count'], config['params'], config['use_quantization']

def initialize_models():
    """
    Initialize the main LLM, search engine, and tokenizer based on user selection.
    
    Args:
        None.     
    Returns:
        Status message
    """


    main_model_name, search_model_name, gpu_count, params, use_quantization = get_params()
    if gpu_count == "Default":
        gpu_count = get_available_gpus()[0]

    status_message = ""
    
    # Initialize main model
    try:
        print(f"Initializing main model: {main_model_name} with {gpu_count} GPUs")
        
        # Import here to avoid loading at startup
        from vllm import LLM, SamplingParams
        
        # Initialize with vLLM
        model_kwargs = {
            "model": main_model_name,
            "tensor_parallel_size": gpu_count,
            "dtype": "half",  # Use half precision for better memory usage
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.5,
            "max_model_len": 32768,
        }
        
        # Add quantization parameter if enabled
        if use_quantization:
            model_kwargs["quantization"] = "AWQ"

        # Initialize the model with all parameters
        model = LLM(**model_kwargs)

        # Set sampling parameters from user inputs
        sampling_params = SamplingParams(
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.95),
            max_tokens=params.get("max_tokens", 512)
        )

        # Store model and sampling parameters globally
        main_model = {
            "model": model,
            "sampling_params": sampling_params,
            "name": main_model_name
        }
        
        status_message += f"Main model '{main_model_name}' initialized successfully\n"
        if use_quantization:
            status_message += f"Quantization: AWQ enabled\n"
    except Exception as e:
        status_message += f"Error initializing main model: {e}\n"
    # Initialize search engine
    try:
        print(f"Initializing search engine with model: {search_model_name}")
        
        # Import your ESGSearchEngine here
        from search_engine.engine import ESGSearchEngine
        
        # Ensure NLTK resources are available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        
        # Determine GPU usage
        use_gpu = gpu_count > 0
        
        # Initialize search engine
        search_engine_instance = ESGSearchEngine(
            model_path=search_model_name,
            tokenizer_model_path=main_model_name,
            use_gpu=use_gpu
        )
        
        # Store search engine
        search_engine = {
            "engine": search_engine_instance,
            "name": search_model_name,
            "params": {
                "top_k": params.get("top_k", 5),
                "rerank_k": params.get("rerank_k", 3),
                "alpha": params.get("alpha", 0.5)
            }
        }
        
        status_message += f"Search engine with model '{search_model_name}' initialized successfully\n"
    except Exception as e:
        status_message += f"Error initializing search engine: {e}\n"
    
    # Initialize tokenizer
    try:
        print(f"Initializing tokenizer: {main_model_name}")
        
        # Initialize tokenizer
        tokenizer_instance = AutoTokenizer.from_pretrained(main_model_name)
        
        # Store tokenizer
        tokenizer = tokenizer_instance
        
        status_message += f"Tokenizer '{main_model_name}' initialized successfully"
    except Exception as e:
        status_message += f"Error initializing tokenizer: {e}"
    
    print(status_message)
    return main_model, search_engine, tokenizer

def process_files():
    """
    Process files in the ./data/pdfs folder.
    
    Args:
        None.
        
    Returns:
        Tuple containing DataFrame of extracted data and path to CSV file
    """
    main_model, search_engine, tokenizer = initialize_models()
    _,_,_,params = get_params()

    # Check if models are initialized
    if main_model is None or search_engine is None or tokenizer is None:
        return pd.DataFrame({"error": ["Please initialize models first"]}), ""
    
    # Extract text from PDFs
    pdf_texts = []
    filenames = []
    all_entries = os.listdir('./data/pdfs')
    files = [os.path.join('./data/pdfs', entry) for entry in all_entries if os.path.isfile(os.path.join('./data/pdfs', entry))]
    for file_path in files:
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        if file_ext == '.pdf':
            # Use the user's provided function to extract text from PDF
            try:
                text_content = extract_text_from_pdf(file_path)
                pdf_texts.append(text_content)
                filenames.append(file_name)
            except Exception as e:
                print(f"Error extracting text from PDF {file_path}: {e}")
        elif file_ext == '.txt':
            # For TXT: Read file directly
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                pdf_texts.append(text_content)
                filenames.append(file_name)
            except Exception as e:
                print(f"Error processing TXT {file_path}: {e}")
    
    if not pdf_texts:
        return pd.DataFrame({"error": ["No valid PDF or TXT files provided"]}), ""
    
    # Process the extracted text
        # Use the extract function if available
    args = type('Args', (), {
        'rerank_k': params.get("rerank_k", 200),
        'top_k': params.get("top_k", 50),
        'alpha': params.get("alpha", 0.7)
    })
    
    results = extract(
        pdf_texts=pdf_texts,
        llm=main_model["model"],
        sampling_params=main_model["sampling_params"],
        tokenizer=tokenizer,
        search_engine=search_engine["engine"],
        args=args
    )

    results["filename"] = filenames
    df = pd.DataFrame.from_dict(results)

    sentiment_results = sentiment_analysis(
        df=df, 
        llm=main_model["model"], 
        sampling_params=main_model["sampling_params"], 
        tokenizer=tokenizer
    )

    df_data, df_score = postprocess(df, sentiment_results)

    file_company_dict = {}
    for i, row in df.iterrows():
        file_company_dict[row['filename']] = row['Company']

    # Convert sentiment analysis results into a DataFrame for CSV output
    sentiment_records = []
    for company_name, fields in sentiment_results.items():
        for extracted_field, details in fields.items():
            sentiment = details.get('sentiment', 'Unknown')
            
            # Match the company name from the fields if needed
            if "Auto Parts-" in company_name:
                match = re.search(r"Auto Parts-(.*?)-\d{4}", company_name)
                if match:
                    company_name = match.group(1).lower()
            elif "Auto manufacturers - Major-" in company_name:
                match = re.search(r"Auto manufacturers - Major-(.*?)-\d{4}", company_name)
                if match:
                    company_name = match.group(1).lower()

            sentiment_records.append([file_company_dict[company_name], extracted_field, sentiment])
    
    df_json_sentiment = pd.DataFrame(sentiment_records, columns=['Company', 'Extracted_field', 'Sentiment'])

    df_data.to_csv('./data/results/esg_extraction_results.csv', index=False)
    df_score.to_csv('./data/results/esg_scoring_results.csv', index=False)
    df_json_sentiment.to_csv('./data/results/sentiment_analysis_results.csv', index=False)

if __name__ == "__main__":
    process_files()