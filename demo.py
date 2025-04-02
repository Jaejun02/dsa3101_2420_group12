from extract import extract, sentiment_analysis
from preprocessing.extract_pdf import extract_text_from_pdf
import gradio as gr
import pandas as pd
import os
import tempfile
from typing import List, Tuple, Dict, Any, Optional
import torch
import json
import numpy as np
from transformers import AutoTokenizer
import re
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

def get_available_models():
    """
    Return list of available models for dropdown.
    
    Returns:
        Dictionary of model types and their available options
    """
    # These are example models - replace with your actual available models
    return {
        "main_models": [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct"
        ],
        "quantized_models": [
            "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            "Qwen/Qwen2.5-3B-Instruct-AWQ",
            "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "Qwen/Qwen2.5-14B-Instruct-AWQ",
            "Qwen/Qwen2.5-32B-Instruct-AWQ",
            "Qwen/Qwen2.5-72B-Instruct-AWQ"
        ],
        "search_models": [
            "BAAI/bge-m3",
        ]
    }


def initialize_models(main_model_name, search_model_name, gpu_count, params, use_quantization=False):
    """
    Initialize the main LLM, search engine, and tokenizer based on user selection.
    
    Args:
        main_model_name: Name of the main model to initialize
        search_model_name: Name of the search model to initialize
        gpu_count: Number of GPUs to use
        params: Dictionary of parameters
        use_quantization: Whether to use quantized models (AWQ)
        
    Returns:
        Status message
    """
    global main_model, search_engine, tokenizer
    
    status_message = ""
    
    # Initialize main model
    try:
        print(f"Initializing main model: {main_model_name} with {gpu_count} GPUs")
        print(f"Using quantization: {'Yes (AWQ)' if use_quantization else 'No'}")
        
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
    
    return status_message


def process_files(files: List[tempfile.NamedTemporaryFile], params: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    """
    Process uploaded files using the ESG extraction logic.
    
    Args:
        files: List of uploaded file objects
        params: Processing parameters
        
    Returns:
        Tuple containing DataFrame of extracted data and path to CSV file
    """
    global main_model, search_engine, tokenizer
    
    # Check if models are initialized
    if main_model is None or search_engine is None or tokenizer is None:
        return pd.DataFrame({"error": ["Please initialize models first"]}), ""
    
    # Extract text from PDFs
    pdf_texts = []
    filenames = []
    
    for file in files:
        file_path = file.name
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

    # qualitative_columns = [
    #     "Narrative on Sustainability Goals and Actions",
    #     "Progress Updates on Emission Reduction Targets",
    #     "Disclosure on Renewable Energy Initiatives and Resource Efficiency Practices",
    #     "Narrative on Workforce Diversity Employee Well-being and Safety",
    #     "Disclosure on Community Engagement and Social Impact Initiatives",
    #     "Narrative on Governance Framework and Board Diversity",
    #     "Disclosure on ESG Risk Management and Stakeholder Engagement",
    #     "Narrative on Innovations in Sustainable Technologies and Product Design",
    #     "Disclosure on Sustainable Supply Chain Management Practices",
    #     "filename"
    # ]
    
    # df_sentiment = df.loc[:, df.columns.intersection(qualitative_columns)]

      sentiment_results = sentiment_analysis(
        df=df, 
        llm=main_model["model"], 
        sampling_params=main_model["sampling_params"], 
        tokenizer=tokenizer
    )
    
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

            sentiment_records.append([company_name, extracted_field, sentiment])
    
    df_json_sentiment = pd.DataFrame(sentiment_records, columns=['Company', 'Extracted_field', 'Sentiment'])
    
    if not df.empty:
        # Save extracted ESG results to CSV
        csv_path = os.path.join(tempfile.gettempdir(), "esg_extraction_results.csv")
        df.to_csv(csv_path, index=False)
        
        # Save sentiment analysis results to CSV
        sentiment_csv_path = os.path.join(tempfile.gettempdir(), "sentiment_analysis_results.csv")
        df_json_sentiment.to_csv(sentiment_csv_path, index=False)
        
        return df, csv_path, df_json_sentiment, sentiment_csv_path
    else:
        return pd.DataFrame({"message": ["No data extracted"]}), ""


def create_gradio_interface():
    """
    Create and launch the Gradio interface.
    """
    # Get available models and GPUs
    available_models = get_available_models()
    available_gpus = get_available_gpus()
    
    with gr.Blocks(title="ESG Document Extractor") as demo:
        gr.Markdown("# ESG Document Extractor")
        gr.Markdown("Upload PDF or TXT files to extract ESG indicators and download results as CSV")
        
        # Model initialization section
        with gr.Tab("Model Configuration"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Quantization toggle
                    use_quantization = gr.Checkbox(
                        label="Use Quantization (AWQ)",
                        value=False,
                        info="Enable model quantization to reduce memory usage and improve performance"
                    )
                    
                    # Model selection
                    main_model_dropdown = gr.Dropdown(
                        choices=available_models["main_models"],
                        value=available_models["main_models"][0] if available_models["main_models"] else None,
                        label="Select Main LLM Model"
                    )
                    
                    search_model_dropdown = gr.Dropdown(
                        choices=available_models["search_models"],
                        value=available_models["search_models"][0] if available_models["search_models"] else None,
                        label="Select Semantic Search Model"
                    )
                    
                    # GPU selection
                    gpu_dropdown = gr.Dropdown(
                        choices=available_gpus,
                        value=available_gpus[0],
                        label=f"Number of GPUs to Use (Detected: {max(available_gpus)})"
                    )
                
                with gr.Column(scale=1):
                    # Parameters table
                    gr.Markdown("### LLM Parameters")
                    
                    # Main model parameters
                    temperature = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature")
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.95, step=0.05, label="Top-p")
                    max_tokens = gr.Slider(minimum=64, maximum=4096, value=512, step=64, label="Max Tokens")
                    
                    # Search parameters - align with ESGSearchEngine.combined_search
                    gr.Markdown("### Search Parameters")
                    top_k = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-k (Results to Return)")
                    rerank_k = gr.Slider(minimum=1, maximum=500, value=200, step=10, label="Rerank-k (Results to Rerank)")
                    alpha = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, label="Alpha (Semantic Search Weight)")
            
            # Initialize button
            initialize_button = gr.Button("Initialize Models", variant="primary")
            model_status = gr.Textbox(value="Models not initialized", label="Model Status", interactive=False)
            
            # Update model dropdown based on quantization toggle
            def update_model_dropdown(use_quant):
                if use_quant:
                    return gr.Dropdown(choices=available_models["quantized_models"], 
                                      value=available_models["quantized_models"][0] if available_models["quantized_models"] else None)
                else:
                    return gr.Dropdown(choices=available_models["main_models"], 
                                      value=available_models["main_models"][0] if available_models["main_models"] else None)
            
            # Connect the toggle to update model dropdown
            use_quantization.change(
                fn=update_model_dropdown,
                inputs=[use_quantization],
                outputs=[main_model_dropdown]
            )
            
            # Initialize models function
            def init_models(main_model_name, search_model_name, gpu_count, temp, top_p_val, max_tok, 
                           top_k_val, rerank_k_val, alpha_val, use_quant):
                params = {
                    "temperature": temp,
                    "top_p": top_p_val,
                    "max_tokens": max_tok,
                    "top_k": top_k_val,
                    "rerank_k": rerank_k_val,
                    "alpha": alpha_val
                }
                return initialize_models(main_model_name, search_model_name, gpu_count, params, use_quant)
            
            # Connect initialization
            initialize_button.click(
                fn=init_models,
                inputs=[
                    main_model_dropdown, search_model_dropdown, gpu_dropdown,
                    temperature, top_p, max_tokens,
                    top_k, rerank_k, alpha, use_quantization
                ],
                outputs=[model_status]
            )
            
        # File processing section
        with gr.Tab("Process Files"):
            with gr.Row():
                with gr.Column(scale=1):
                    # File upload component
                    file_input = gr.File(
                        file_types=[".pdf", ".txt"],
                        file_count="multiple",
                        label="Upload Files (PDF or TXT)"
                    )
                    
                    # Process parameters
                    gr.Markdown("### Processing Parameters")
                    process_top_k = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-k")
                    process_rerank_k = gr.Slider(minimum=1, maximum=500, value=200, step=10, label="Rerank-k")
                    process_alpha = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, label="Alpha")
                    
                    # Process button
                    process_button = gr.Button("Extract ESG Data", variant="primary")
                
                with gr.Column(scale=2):
                    # Output components
                    output_display = gr.Dataframe(label="Extracted ESG Data", interactive=False)
                    download_button = gr.File(label="Download CSV")
                    # Output Sentiment Analysis components
                    sentiment_output_display = gr.Dataframe(label="Sentiment Analysis Results", interactive=False) 
                    sentiment_download_button = gr.File(label="Download Sentiment Analysis CSV")  
            
            # Set up processing logic
            def process_and_display(files, top_k_val, rerank_k_val, alpha_val):
                if not files:
                    return pd.DataFrame({"message": ["No files uploaded"]}), None, None, None
                
                # Check if model is initialized
                global main_model, search_engine, tokenizer
                if main_model is None or search_engine is None or tokenizer is None:
                    return pd.DataFrame({"message": ["Please initialize models first"]}), None, None, None
                
                # Gather parameters
                params = {
                    "top_k": top_k_val,
                    "rerank_k": rerank_k_val,
                    "alpha": alpha_val
                }
                
                # Update search parameters
                if search_engine:
                    search_engine["params"] = params
                
                # Process files
                df, csv_path, df_sentiment_results, sentiment_csv_path = process_files(files, params)
                
                if not csv_path:
                    return df, None, None, None
                
                return df, csv_path, df_sentiment_results, sentiment_csv_path
            
            # Connect components
            process_button.click(
                fn=process_and_display,
                inputs=[
                    file_input,
                    process_top_k, process_rerank_k, process_alpha
                ],
                outputs=[output_display, download_button, sentiment_output_display, sentiment_download_button]
            )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    # Configure server to be accessible remotely
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    
