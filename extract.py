from vllm import LLM, SamplingParams
from search_engine.engine import ESGSearchEngine
import json
import re
import numpy as np
import pandas as pd


with open("./prompts/esg_indicators.json", 'r') as file:
    esg_indicators = json.load(file)

INDICATORS = list(esg_indicators.keys())

with open('./prompts/esg_context.json', 'r') as file:
    ESG_CONTEXT = json.load(file)

KEYWORDS = []
[KEYWORDS.extend(indicators['keywords']) for indicators in esg_indicators.values()]
KEYWORDS = list(set(KEYWORDS))

with open("./prompts/oneshots.json", 'r') as file:
    examples = json.load(file)
    
with open('./prompts/sentiment_context.json', 'r') as file:
    sentiment_analysis = json.load(file)


def extract(pdf_texts, llm, sampling_params, tokenizer, search_engine, args):
    all_res = {}
    
    for cur_indicator in INDICATORS:
        input_texts = []
        res = []
        semantic_query = "What is the " + cur_indicator + "?" + "Here are some keywords to help you: " + ", ".join(esg_indicators[cur_indicator]['keywords'])
        keyword_query_list = esg_indicators[cur_indicator]['keywords']

        for pdf_text in pdf_texts:
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            search_engine.initialize(pdf_text)
            results, _ = search_engine.combined_search(semantic_query, keyword_query_list, rerank_k = args.rerank_k, top_k=args.top_k, alpha=args.alpha)
            
            text_input = ""
            for text in results:
                text_input += text
                text_input += "\n\n"
            msg = f"""You are an expert ESG reporting analyst specializing in the automotive industry. Your task is to extract '{cur_indicator}' indicator given a text to extract from.

                Instructions:
                """
            if esg_indicators[cur_indicator]['data_type'] == "Quantitative":
                msg += f"""        1. The current indicator is a quantitative indicator. You must extract relevant numeric data or a short string that includes one of the possible units specified in the "unit" instruction.
                2. The allowed list of units are: {esg_indicators[cur_indicator]['unit']}. For any other units found, convert into one of the units in the list.
                """
            else:
                msg += f"""        1. The current indicator is a qualitative indicator. You must locate the sentence(s) that best describe the indicator within the text to extract.
                2. If the relevant information is spread across disjoint sentences, return them as a separate element of a list; otherwise, return a list with single sentence.
                """
            msg += f"""        3. The following key words: {esg_indicators[cur_indicator]['keywords']} are some words that you can watch out for in extraction, but consider the overall context to ensure accurate extraction.
            4. You must follow the additional instructions specified here: {esg_indicators[cur_indicator]['extraction_notes']}.
            5. The final output should be enclosed with <output> and </output> tags.
            6. If there are no data to be found, return <output>No data available for {cur_indicator}</output>.

            The following are some contexts that you may refer to when understanding the text to extract from:
            Specific to '{cur_indicator}': {esg_indicators[cur_indicator]['background']}
            General ESG background: {ESG_CONTEXT}

            Example:
            {examples[cur_indicator]}.

            Here is the text to extract '{cur_indicator}' from:
            '''
            {text_input}
            ''''
            """
            messages = [{'role': 'user', 'content': msg}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_texts.append(input_text)
            
        outputs = llm.generate(input_texts, sampling_params=sampling_params)
        for i, output in enumerate(outputs):
            op = output.outputs[0].text
            found = re.findall(r'<output>(.*?)</output>', op)
            
            if found:
                op = found[0]
            else:
                op = None
                
            res.append(op)
        
        all_res[cur_indicator] = res
        
    return all_res


def sentiment_analysis(df, llm, sampling_params, tokenizer, args):
    df_big = df.map(lambda x: np.nan if isinstance(x, str) and "No data available" in x else x)
    qualitative_columns = sentiment_analysis.keys()
    input_texts = []
    identifiers = []
    for row in df_big.iterrows():
        for col in qualitative_columns:
            if col in row[1]:
                text = row[1][col]
                if not pd.isna(text):
                    msg = f"""
                    Using the following background information and examples, perform a sentiment analysis on the provided sentence that is classified as relevant to the ESG topic of {col}.
                    
                    Background:
                    'This analysis focuses on evaluating ESG reports, where the accuracy and integrity of the environmental, social, and governance disclosures are critical. Analysts assess not only the reported data but also the credibility of the claims, the robustness of the reporting metrics, and potential signals of greenwashing. Key considerations include environmental performance (carbon emissions, energy efficiency, waste management), social factors (labor practices, diversity and inclusion, community engagement), and governance aspects (board structure, transparency, risk management).'
                    {sentiment_analysis[col]['background']}
                    
                    Few-shot Examples:
                    {sentiment_analysis[col]['example']}
                    
                    Instructions:
                    1. Analyze the sentiment of the given sentence using SMART analysis (Specific, Measurable, Attainable, Relevant, and Timely) as your guiding framework.
                    2. Incorporate common ESG analysis factors, such as:
                        - Environmental: Assess claims regarding carbon emissions, renewable energy, waste management, and other environmental impacts.
                        - Social: Consider aspects like labor practices, diversity and inclusion, community engagement, and overall social responsibility.
                        - Governance: Evaluate transparency, board structure, risk management, and the credibility of reported data.
                    3. Carefully identify any potential signs of greenwashing, where the narrative might exaggerate or mislead regarding actual ESG performance.
                    4. Based on the analysis, determine if the sentiment of the sentence is "positive", "negative", or "neutral".
                    5. Return only one of these outputs, and enclose your result strictly within <output> and </output> tags. For example, if the analysis is positive, your output should be: <output>positive</output>.
                    
                    Provide only the sentiment analysis result with the required tags.
                    
                    Here is the input sentence you must sentiment analyze on: {text}
                    """
                    messages = [{'role': 'user', 'content': msg}]
                    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    input_texts.append(input_text)
                    identifiers.append((row[1]["filename"], col, text))
                
    dict = {}
    outputs = llm.generate(input_texts, sampling_params=sampling_params)
    for i, output in enumerate(outputs):
        found = re.findall(r'<output>(.*?)</output>', output.outputs[0].text)
        if found:
            op = found[0]
        else:
            continue
        
        filename, column, text = identifiers[i]
        if filename not in dict:
            dict[filename] = {}
            dict[filename][column] = {"original_text": text, "sentiment": op}
        else:
            dict[filename][column] = {"original_text": text, "sentiment": op}
        
    return dict
    