import os
import requests
from tqdm.auto import tqdm


url_file = "../data/pdf_reports_urls.txt"
output_dir = "../data/pdfs"
os.makedirs(output_dir, exist_ok=True)

def download_pdf(url, directory, idx):
    # Parse the URL to extract the filename.
    filename = f"pdf_{idx}.pdf"  # Default filename if extraction fails.
    
    # Build the full file path.
    file_path = os.path.join(directory, filename)
    
    try:
        # Send a GET request to download the PDF.
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors.
        
        # Write the PDF content to the file in binary mode.
        with open(file_path, 'wb') as file:
            file.write(response.content)
        
        print(f"PDF downloaded and saved to: {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        
        
def main():
    with open(url_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f.readlines() if line.strip()]

    for idx, url in tqdm(enumerate(urls), total=len(urls)):
        download_pdf(url, output_dir, idx)
        
        
if __name__ == "__main__":
    main()