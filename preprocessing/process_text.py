from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
import PIL
import re
import os
import glob

text_dir = ""
output_dir = "../data/preprocessed"
os.makedirs(output_dir, exist_ok=True)


def replace_images_with_descriptions(markdown_text, descriptions, pattern):
    """
    Replace markdown image references with provided descriptions.

    Parameters:
        markdown_text (str): The markdown text containing image syntax.
        descriptions (List[str]): A list of image descriptions in the order the images appear.
    
    Returns:
        str: The updated markdown text with image descriptions replacing the image syntax.
    """
    # Create an iterator over the descriptions list
    description_iter = iter(descriptions)
    
    def replacer(match):
        try:
            # Get the next description from the list
            desc = next(description_iter)
        except StopIteration:
            # In case there are more images than descriptions
            desc = "[No description provided]"
        return desc

    # Replace each occurrence of the image markdown with its corresponding description
    updated_markdown = re.sub(pattern, replacer, markdown_text)
    return updated_markdown


def main():
    txt_files = glob.glob(f'{output_dir}/*.txt')
    print(txt_files)
    
    vlm = LLM(model="llava-hf/llava-1.5-7b-hf", )
    sampling_params = SamplingParams(max_tokens=150)
    prompt = "USER: <image>\Describe the content of the image. If the image contains data (such as tables, charts, graphs, or numerical information) related to ESG, provide a detailed description that includes all the data. If the image does not contain data, provide a simple one-line description.\nASSISTANT:"
    image_pattern = r'!\[\]\((.*?)\)'
    
    for txt_file in txt_files:
        extracted_text = open(txt_file, 'r').read()
        images = re.findall(image_pattern, extracted_text)
        inputs = [
            {
                "prompt": prompt, 
                "multi_modal_data": {"image": PIL.Image.open(image)}
            } for image in images
        ]
        outputs = vlm.generate(inputs, sampling_params)
        image_descriptions = ["[Image Description:" + output.outputs[0].text + "]" for output in outputs]
        processed_text = replace_images_with_descriptions(extracted_text, image_descriptions, image_pattern)
        
        with open(".." + txt_file.split(".")[-2] + "_processed.txt", "w", encoding="utf-8") as txt_file:
            txt_file.write(processed_text)

       
if __name__ == "__main__":
    main()