import os
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from tqdm.auto import tqdm
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
        
    md_content = pipe_result.get_markdown(local_image_dir)
    return md_content


def extract_text_from_pdf(pdf_path):
    reader = FileBasedDataReader("")
    pdf_bytes = reader.read(pdf_path)
    extracted_text = extract_text_from_pdf_bytes(pdf_bytes, pdf_path)
    return extracted_text


# def main():
#     reader = FileBasedDataReader("")
#     pdfs = os.listdir(pdf_dir)

#     for idx, pdf_file_name in tqdm(enumerate(sorted(pdfs)), total=len(pdfs)):
#         name_without_suff = pdf_file_name.split(".")[0]
#         txt_output_path = os.path.join(output_dir, f"{name_without_suff}.txt")
#         file_path = os.path.join(pdf_dir, pdf_file_name)

#         pdf_bytes = reader.read(file_path)
#         extracted_text = extract_text_from_pdf_bytes(pdf_bytes, name_without_suff)
        
#         # Save extracted text into its own file
#         with open(txt_output_path, "w", encoding="utf-8") as txt_file:
#             txt_file.write(extracted_text)

#         # except Exception as e:
#         #     print(f"Error processing {pdf_file_name}: {e}")

#     print("Processing complete")

# if __name__ == "__main__":
#     main()
