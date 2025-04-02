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

main_model = None
search_engine = None
tokenizer = None

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
    with open('./data/config.json') as file:
        config = json.load(file)
    return config.values()

def initialize_models():
    """
    Initialize the main LLM, search engine, and tokenizer based on user selection.
    
    Args:
        None        
    Returns:
        Status message
    """
    global main_model, search_engine, tokenizer
    
    main_model_name, search_model_name, gpu_count, params = get_all_params()
    if gpu_count == "Default":
        gpu_count = get_available_gpus()

    status_message = ""
    
    # Initialize main model
    try:
        print(f"Initializing main model: {main_model_name} with {gpu_count} GPUs")
        
        # Import here to avoid loading at startup
        from vllm import LLM, SamplingParams
        
        # Initialize with vLLM
        model = LLM(
            model=main_model_name,
            tensor_parallel_size=gpu_count,
            dtype="half",  # Use half precision for better memory usage
            trust_remote_code=True,
            gpu_memory_utilization=0.5,
            max_model_len=32768,
            # enforce_eager=True,  # Enforce eager mode to avoid CUDA graphs which use signals
            # disable_custom_all_reduce=True if gpu_count > 1 else False  # Disable custom all-reduce for multi-GPU
        )
        
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
    except Exception as e:
        status_message += f"Error initializing main model: {e}\n"
    
    # Initialize search engine
    try:
        print(f"Initializing search engine with model: {search_model_name}")
        
        # Import your ESGSearchEngine here
        from transformers import AutoTokenizer
        
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
    
    return status_message

def process_files():
    """
    Process files in the ./data folder.
    
    Args:
        Nonbe.
        
    Returns:
        Tuple containing DataFrame of extracted data and path to CSV file
    """
    global main_model, search_engine, tokenizer
    
    _,_,_,params = get_params()

    # Check if models are initialized
    if main_model is None or search_engine is None or tokenizer is None:
        return pd.DataFrame({"error": ["Please initialize models first"]}), ""
    
    # Extract text from PDFs
    pdf_texts = []
    filenames = []
    all_entries = os.listdir('./data')
    files = [entry for entry in all_entries if os.path.isfile(os.path.join('./data', entry))]
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

    qualitative_columns = [
        "Narrative on Sustainability Goals and Actions",
        "Progress Updates on Emission Reduction Targets",
        "Disclosure on Renewable Energy Initiatives and Resource Efficiency Practices",
        "Narrative on Workforce Diversity Employee Well-being and Safety",
        "Disclosure on Community Engagement and Social Impact Initiatives",
        "Narrative on Governance Framework and Board Diversity",
        "Disclosure on ESG Risk Management and Stakeholder Engagement",
        "Narrative on Innovations in Sustainable Technologies and Product Design",
        "Disclosure on Sustainable Supply Chain Management Practices"
    ]
    
    df_sentiment = df.loc[:, df.columns.intersection(qualitative_columns)]

    sentiment_results = sentiment_analysis(
        df=df_sentiment, 
        llm=main_model["model"], 
        sampling_params=main_model["sampling_params"], 
        tokenizer=tokenizer
    )

    with open('./data/sentiment_analysis_results.json', 'w') as file:
        json.dump(sentiment_results, file, indent=4)

    postprocess(df, sentiment_results)

if __name__ == "__main__":
    initialize_models()
    process_files()