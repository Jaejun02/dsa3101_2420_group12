import os
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from tqdm.auto import tqdm
# from vllm import LLM
# import PIL
import re


pdf_dir = "../data/pdfs"
output_dir = "../data/preprocessed"
os.makedirs(output_dir, exist_ok=True)
    
def extract_text_from_pdf_bytes(pdf_bytes, doc_name):
    local_image_dir = os.path.join(output_dir, doc_name, "images")
    image_writer = FileBasedDataWriter(local_image_dir)
    # md_writer = FileBasedDataWriter(output_dir)

    ds = PymuDocDataset(pdf_bytes)

    if ds.classify() == SupportedPdfParseMethod.OCR:
        print("Used OCR")
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        print("Used Text Parsing")
        infer_result = ds.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    # infer_result.draw_model(os.path.join(output_dir, f"{doc_name}_model.pdf"))
    # pipe_result.draw_layout(os.path.join(output_dir, f"{doc_name}_layout.pdf"))
    # pipe_result.draw_span(os.path.join(output_dir, f"{doc_name}_spans.pdf"))
    # pipe_result.dump_md(md_writer, f"{doc_name}.md", local_image_dir)
    md_content = pipe_result.get_markdown(local_image_dir)
    return md_content


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
    reader = FileBasedDataReader("")
    pdfs = os.listdir(pdf_dir)
    # vlm = LLM(model="llava-hf/llava-1.5-7b-hf")
    # prompt = "USER: <image>\Describe the content of the image. If the image contains data (such as tables, charts, graphs, or numerical information), provide a detailed description that includes all the data. If the image does not contain data, provide a simple one-line description.\nASSISTANT:"

    for idx, pdf_file_name in tqdm(enumerate(sorted(pdfs)), total=len(pdfs)):
        name_without_suff = pdf_file_name.split(".")[0]
        txt_output_path = os.path.join(output_dir, f"{name_without_suff}.txt")
        file_path = os.path.join(pdf_dir, pdf_file_name)
        # image_pattern = r'!\[\]\((.*?)\)'

        try:
            pdf_bytes = reader.read(file_path)
            extracted_text = extract_text_from_pdf_bytes(pdf_bytes, name_without_suff)
            # vlm = LLM(model="llava-hf/llava-1.5-7b-hf")
            # images = re.findall(image_pattern, extracted_text)
            # inputs = [
            #     {
            #         "prompt": prompt, 
            #         "multi_modal_data": {"image": PIL.Image.open(image)}
            #     } for image in images
            # ]
            # outputs = vlm.generate(inputs)
            # image_descriptions = ["[" + output.outputs[0].txt + "]" for output in outputs]
            # processed_text = replace_images_with_descriptions(extracted_text, image_descriptions, image_pattern)
            # print(processed_text)

            # Save extracted text into its own file
            with open(txt_output_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(extracted_text)

            # print(f"Saved extracted text to {txt_output_path}\n")

        except Exception as e:
            print(f"Error processing {pdf_file_name}: {e}")

    print("Processing complete")

if __name__ == "__main__":
    main()
