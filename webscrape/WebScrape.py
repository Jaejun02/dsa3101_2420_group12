import sys
import os
import logging
from pathlib import Path

try:
    from ResponsibilityReports import ResponsibilityReports
except ImportError:
    current_dir = Path(__file__).parent
    lower_dir = current_dir / "webscrape"

    sys.path.append(str(lower_dir))
    try:
        from ResponsibilityReports import ResponsibilityReports
    except ImportError:
        raise ImportError("Could not import ResponsibilityReports.py\
                            in either the current directory or the 'webscrape' subdirectory")
    
def setup_logger():
    logger = logging.getLogger("ESGReportScraper")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger

def main():
    logger = setup_logger()
    logger.info("Starting ESG Report Scraper")
    
    data_dir = Path("../data")
    if not data_dir.exists():
        logger.info(f"Creating data directory at {data_dir.resolve()}")
        data_dir.mkdir(parents=True)
    
    logger.info("Initializing ResponsbilityReports Scraper")
    scraper = ResponsibilityReports()

    logger.info("Starting report download process")
    result, summary = scraper.download_reports(scrape_url=False)

    if result:
        logger.info("Successfully completed the ESG report extraction process.")
    else:
        logger.info("Failed to cmoplete the ESG report extraction process.")
    
    return summary

if __name__ == "__main__":
    main()