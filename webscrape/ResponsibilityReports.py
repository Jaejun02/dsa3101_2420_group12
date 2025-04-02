import requests
from bs4 import BeautifulSoup
import re
import os
import logging

class ResponsibilityReports:
    """
    This class creates a webscraper to download ESG Reports from responsibilityreports.com.
    The class has the following functions:
        - webscrape_report_urls: Scrapes the website to find URLs to download ESG Reports.
        - download_helper: Downloads the PDF from the given URL and saves it to the output path. Used as a helper
                           function of download_reports.
        - download_reports: Downloads the ESG Reports from the URLs obtained from webscrape_report_urls function.
    """
    DATA_DIR = "../data"
    RESPONSIBILITY_REPORTS_BASE = "https://www.responsibilityreports.com/"

    def __init__(self):
        # Setting up logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def webscrape_report_urls(self):
        """
        This function scrapes the responsibilityreports.com website to find Industries that 
        falls under Automotive Industries.
        Args: 
            None.
        Returns:
            download_urls -> List[String]: List of URLs to download ESG Reports.
            download_dests -> List[String]: List of destinations to save the downloaded ESG Reports.
        """
        # Step 1: Get List of All Industries (To Find Industries under Automotive Industry)
        response = requests.get(self.RESPONSIBILITY_REPORTS_BASE + "Browse/Industry")
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Step 2: Filter Industry Names and Links under Automotive Industry
        industry_link = soup.find_all("a", href=re.compile(r'/Companies\?ind=i\d'))
        industry_link = [(i['href'], i.text.strip()) for i in industry_link]
        link, industry = zip(*industry_link)
        automotive_url, automotive_industry = [], []
        for i, ind in enumerate(industry):
            if ind in ["Auto Manufacturers - Major", "Auto Parts", "Auto Dealerships", "Auto Parts Stores", "Auto Parts Wholesale", "Trucks & Other Vehicles"]:
                automotive_url.append(self.RESPONSIBILITY_REPORTS_BASE.strip("/") + link[i])
                automotive_industry.append(ind)
        self.logger.debug("Scraped auotomotive industries urls: %s", automotive_url)

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
            company_link = [self.RESPONSIBILITY_REPORTS_BASE + i for i in company_link]
            for j, company in enumerate(company_link):
                response = requests.get(company)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                report = soup.find('a', href=re.compile(r'/Click/\d+'))
                if report:
                    report_link, report_name = report['href'], report.text.strip()
                    report_link = self.RESPONSIBILITY_REPORTS_BASE + report_link
                    download_urls.append(report_link)
                    download_dests.append(self.DATA_DIR + f"/{automotive_industry[i]}-{company_name[j]}-{report_name}.pdf")
        self.logger.debug("Found %d reports to download", len(download_urls))
        return download_urls, download_dests

    def download_helper(self, url, output_path):
        """
        This function downloads the PDF from the given single URL and saves it to the output path.
        It additionally logs the Industry and Company names, and url to a text file.
        Args:
            url -> String: URL to download the PDF.
            output_path -> String: Path to save the downloaded PDF.
        Returns:
            Success -> boolean: True if the PDF is downloaded successfully, False otherwise.
        """
        self.logger.debug("Downloading PDF from URL: %s", url)
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.debug("Created directory: %s", output_dir)
            response = requests.get(url)
            response.raise_for_status()
            with open(output_path, 'wb') as pdf_file:
                pdf_file.write(response.content)
            # with open(self.DATA_DIR + "/pdf_reports_paths.txt", 'a') as file:
            #     file.write(output_path + '\n')
            # with open(self.DATA_DIR + "/pdf_reports_urls.txt", 'a') as file:
            #     file.write(url+'\n')
            return True
        except requests.RequestException as e:
            self.logger.error("Error downloading the PDF %s: %s", output_path, e)
            return False
        except IOError as e:
            self.logger.error("Error saving the PDF %s: %s", output_path, e)
            return False

    def download_reports(self, scrape_url = False):
        """
        This function downloads the ESG Reports from the URLs obtained from webscrape_report_urls function.
        Args:
            None.
        Returns:
            Result -> String: Summary statement of the download process.
        """
        self.logger.debug("Starting download of reports.")
        if scrape_url:
            download_urls, download_dests = self.webscrape_report_urls()
        else:
            download_urls = [line.strip() for line in open(self.DATA_DIR + "/pdf_reports_urls.txt", 'r')]
            download_dests = [line.strip() for line in open(self.DATA_DIR + "/pdf_reports_paths.txt", 'r')]
        results = []
        for i, url in enumerate(download_urls):
            result = self.download_helper(url, download_dests[i])
            results.append(result)
        summary = f"\rSuccessfully Downloaded {sum(results)}/{len(results)} PDFs"
        self.logger.info(summary)
        return True, summary