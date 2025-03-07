import requests
from bs4 import BeautifulSoup
import re
import os
from urllib.parse import urljoin
import shutil

DATA_DIR = "../data"
RESPONSIBILITY_REPORTS_BASE = "https://www.responsibilityreports.com/"
SUSTAINABILITY_REPORTS_BASE = "https://www.sustainabilityreports.com/"

def webscrape_report_urls():
    """
    This function scrapes the responsbilityreports.com website to find Industries that 
    falls under Automotive Industries.
    Input: 
        None.
    Returns:
        download_urls -> List[String]: List of URLs to download ESG Reports.
        download_dests -> List[String]: List of destinations to save the downloaded ESG Reports.
    """
    # Step 1: Get List of All Industries (To Find Industries under Automotive Industry)
    response = requests.get(RESPONSIBILITY_REPORTS_BASE + "Browse/Industry")
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    # Step 2: Filter Industry Names and Links under Automotive Industry
    industry_link = soup.find_all("a", href=re.compile(r'/Companies\?ind=i\d'))
    industry_link = [(i['href'], i.text.strip()) for i in industry_link]
    link, industry = zip(*industry_link)
    automotive_url, automotive_industry = [], []
    for i, ind in enumerate(industry):
        if ind in ["Auto Manufacturers - Major", "Auto Parts", "Auto Dealerships", "Auto Parts Stores", "Auto Parts Wholesale", "Trucks & Other Vehicles"]:
            automotive_url.append(RESPONSIBILITY_REPORTS_BASE.strip("/") + link[i])
            automotive_industry.append(ind)

    # Step 3: Retrieve ESG Report URLs for Companies under Automotive Industry
    download_urls = []
    download_dests = []
    for i, url in enumerate(automotive_url):
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        companies = soup.find_all("a", href=re.compile(r'/Company/[a-z\\-]+'))
        companies = [(i['href'], i.text.strip()) for i in companies]
        company_link, company_name = zip(*companies)
        company_link = [RESPONSIBILITY_REPORTS_BASE + i for i in company_link]
        for j, company in enumerate(company_link):
            response = requests.get(company)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            report = soup.find('a', href=re.compile(r'/Click/\d+'))
            if report:
                report_link, report_name = report['href'], report.text.strip()
                report_link = RESPONSIBILITY_REPORTS_BASE + report_link
                download_urls.append(report_link)
                download_dests.append(DATA_DIR + f"/{automotive_industry[i]}/{company_name[j]}/{report_name}.pdf")
    return download_urls, download_dests

def download_helper(url, output_path):
    """
    This function downloads the PDF from the given single URL and saves it to the output path.
    It additionally logs the Industry and Company names, and url to a text file.
    Input:
        url -> String: URL to download the PDF.
        output_path -> String: Path to save the downloaded PDF.
    Returns:
        Success -> boolean: True if the PDF is downloaded successfully, False otherwise.
    """
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, 'wb') as pdf_file:
            pdf_file.write(response.content)
        with open(DATA_DIR + "/ResponsbilityReports.txt", 'a') as file:
            file.write(f"Industry: {output_path.split("/")[2]}, Company: {output_path.split("/")[3]}\n")
        with open(DATA_DIR + "/pdf_reports_urls.txt", 'a') as file:
            file.write(url+'\n')
        return True
    except requests.RequestException as e:
        print(f"Error downloading the PDF: {e}")
        return False
    except IOError as e:
        print(f"Error saving the PDF: {e}")
        return False

def download_reports():
    """
    This function downloads the ESG Reports from the URLs obtained from webscrape_report_urls function.
    Input:
        None.
    Returns:
        Result -> String: Summary statement of the download process.
    """
    download_urls, download_dests = webscrape_report_urls()
    results = []
    for i, url in enumerate(download_urls):
        results.append(download_helper(url, download_dests[i]))
    return f"\rSuccessfully Downloaded {sum(results)}/{len(results)} PDFs"