import pandas as pd
import numpy as np
import re
import os

def transform_row(row):
    row = row.map(lambda x: np.nan if isinstance(x, str) and "No data available" in x else x)
    row = row.fillna("")

    # Total Energy Consumption - kwh, mwh, gwh, gj + thousand, million -> mwh
    units = re.findall(r'\d[\d,\s.]+([A-Za-z\s]+)', row['Total Energy Consumption in Production'])
    num = re.findall(r'(\d[\d,\s.]+)[A-Za-z\s]+',row['Total Energy Consumption in Production'])
    if units and num:
        if len(units) > 1 and re.sub(" ","",units[0]) == "":
                units = units[1].split()
        else:
            units = units[0].split()
        tmp = 0
        for n in num:
            tmp += float(n.replace(",","").replace(" ", ""))
        num = tmp / len(num)
        mult, unit = '', ''
        if len(units) > 1:
            mult, unit = units[0].lower(), units[1].lower()
        else:
            unit = units[0].lower()
        if mult not in ['thousand', 'million', 'billion'] and unit not in ['kwh', 'mwh', 'gwh', 'gj']:
            num = np.nan
        else:
            if mult == 'thousand':
                num *= 1000
            if mult == 'million':
                num *= 1000000
            if mult == 'billion':
                num *= 1000000000
            if unit == 'kwh':
                num /= 1000.0
            if unit == 'gwh':
                num *= 1000.0
            if unit == 'gj':
                num /= 3.6
    else:
        num = np.nan
    row['Total Energy Consumption in Production'] = num

    # Energy Consumption per Vehicle Production - kwh/vehicle, mwh/vehicle, mj/vehicle -> mwh/vehicle
    units = re.findall(r'\d[\d,\s.]+([A-Za-z\s/]+)', row['Energy Consumption per Vehicle Production'])
    num = re.findall(r'(\d[\d,\s.]+)[A-Za-z\s/]+', row['Energy Consumption per Vehicle Production'])
    if units and num:
        if len(units) > 1 and re.sub(" ","",units[0]) == "":
                units = units[1].split()
        else:
            units = units[0].split()
        tmp = 0
        for n in num:
            tmp += float(n.replace(",","").replace(" ", ""))
        num = tmp / len(num)
        mult, unit = '', ''
        if len(units) > 1:
            mult, unit = units[0].lower(), units[1].lower()
        else:
            unit = units[0].lower()
        if mult not in ['thousand', 'million', 'billion'] and unit not in ['kwh', 'mwh', 'gwh', 'gj']:
            num = np.nan
        else:
            if mult == 'thousand':
                num *= 1000
            if mult == 'million':
                num *= 1000000
            if mult == 'billion':
                num *= 1000000000
            if unit == 'kwh/vehicle':
                num /= 1000.0
            if unit == 'mj/vehicle':
                num *= 3600
            if num < 1.5:
                num = np.nan
    else:
        num = np.nan
    row['Energy Consumption per Vehicle Production'] = num

    # Total Water Usage - m³, megaliters, gallons, kl, metric tons, ML + thousand, million -> m³
    units = re.findall(r'\d[\d,\s.]+([A-Za-z\s³]+)', row['Total Water Usage'])
    num = re.findall(r'(\d[\d,\s.]+)[A-Za-z\s³]+', row['Total Water Usage'])
    if units and num:
        if len(units) > 1 and re.sub(" ","",units[0]) == "":
                units = units[1].split()
        else:
            units = units[0].split()
        tmp = 0
        for n in num:
            tmp += float(n.replace(",","").replace(" ", ""))
        num = tmp / len(num)
        mult, unit = '', ''
        if len(units) > 1:
            mult, unit = units[0].lower(), units[1].lower()
        else:
            unit = units[0].lower()
        if mult not in ['thousand', 'million', 'billion'] and unit not in ['m³', 'megaliters', 'gallons', 'kl', 'metric tons', 'ml']:
            num = np.nan
        else:
            if mult == 'thousand':
                num *= 1000
            if mult == 'million':
                num *= 1000000
            if mult == 'billion':
                num *= 1000000000
            if unit == 'megaliters' or unit == 'ml':
                num *= 1000
            if unit == 'gallons':
                num *= 0.00378541
    else:
        num = np.nan
    row['Total Water Usage'] = num

    # Total Wastewater Volume Generated - m³, gallons, liters -> m³
    units = re.findall(r'\d[\d,\s.]+([A-Za-z\s³]+)', row['Total Wastewater Volume Generated'])
    num = re.findall(r'(\d[\d,\s.]+)[A-Za-z\s³]+', row['Total Wastewater Volume Generated'])
    if units and num:
        if len(units) > 1 and re.sub(" ","",units[0]) == "":
                units = units[1].split()
        else:
            units = units[0].split()
        tmp = 0
        for n in num:
            tmp += float(n.replace(",","").replace(" ", ""))
        num = tmp / len(num)
        mult, unit = '', ''
        if len(units) > 1:
            mult, unit = units[0].lower(), units[1].lower()
        else:
            unit = units[0].lower()
        if mult not in ['thousand', 'million', 'billion'] and unit not in ['m³', 'gallons', 'liters']:
            num = np.nan
        else:
            if mult == 'thousand':
                num *= 1000
            if mult == 'million':
                num *= 1000000
            if mult == 'billion':
                num *= 1000000000
            if unit == 'liters':
                num *= 0.001
            if unit == 'gallons':
                num *= 0.00378541
    else:
        num = np.nan
    row['Total Wastewater Volume Generated'] = num

    # Water Recycling and Reuse Rate % -> ratio
    tmp = re.sub(" ", "", re.sub("\\\\","",re.sub("[$]","",row['Water Recycling and Reuse Rate'])))
    if '%' in tmp:
        num = np.mean(np.array(re.findall(r"([\d.]+)%", tmp)).astype('float'))/100.0
    else:
        num = np.nan
    if num >= 1.5:
        num = np.nan
    row['Water Recycling and Reuse Rate'] = np.min([num, 1.00])

    # Total GHG Emissions - Metric Tons, T, Kton, KG -> Metric Tons
    units = re.findall(r'\d[\d,\s.]+([A-Za-z\s³]+)', row['Total GHG Emissions'])
    num = re.findall(r'(\d[\d,\s.]+)[A-Za-z\s³]+', row['Total GHG Emissions'])
    if units and num:
        if len(units) > 1 and re.sub(" ","",units[0]) == "":
                units = units[1].split()
        else:
            units = units[0].split()
        tmp = 0
        for n in num:
            tmp += float(n.replace(",","").replace(" ", ""))
        num = tmp / len(num)
        mult, unit = '', ''
        if units[0] in ['thousand', 'million', 'billion']:
            mult = units[0].lower()
            unit = ' '.join(units[1:]).lower()
        else:
            unit = ' '.join(units).lower()
        if mult not in ['thousand', 'million', 'billion'] and unit not in ['metric tons co', 'kg co', 'kton co', 'tonnes co']:
            num = np.nan
        else:
            if mult == 'thousand':
                num *= 1000
            if mult == 'million':
                num *= 1000000
            if mult == 'billion':
                num *= 1000000000
            if unit == 'kg co':
                num *= 1000
            if unit == 'kton co':
                num *= 0.001
    else:
        num = np.nan
    row['Total GHG Emissions'] = num

    # GHG Emissions and Intensity per Vehicle - T, Kg -> T
    units = re.findall(r'\d[\d,\s.]+([A-Za-z\s³]+)', row['GHG Emissions and Intensity per Vehicle'])
    num = re.findall(r'(\d[\d,\s.]+)[A-Za-z\s³]+', row['GHG Emissions and Intensity per Vehicle'])
    if units and num:
        if len(units) > 1 and re.sub(" ","",units[0]) == "":
                units = units[1].split()
        else:
            units = units[0].split()
        tmp = 0
        for n in num:
            tmp += float(n.replace(",","").replace(" ", ""))
        num = tmp / len(num)
        mult, unit = '', ''
        if units[0] in ['thousand', 'million', 'billion']:
            mult = units[0].lower()
            unit = ' '.join(units[1:]).lower()
        else:
            unit = ' '.join(units).lower()
        if mult not in ['thousand', 'million', 'billion'] and unit not in ['t co', 'kg co']:
            num = np.nan
        else:
            if mult == 'thousand':
                num *= 1000
            if mult == 'million':
                num *= 1000000
            if mult == 'billion':
                num *= 1000000000
            if unit == 'kg co':
                num *= 0.001
    else:
        num = np.nan
    row['GHG Emissions and Intensity per Vehicle'] = num

    # Total Manufacturing Waste Generation - Metric Tons, Tons, Kg -> Metric Tons
    units = re.findall(r'\d[\d,\s.]+([A-Za-z\s³]+)', row['Total Manufacturing Waste Generation'])
    num = re.findall(r'(\d[\d,\s.]+)[A-Za-z\s³]+', row['Total Manufacturing Waste Generation'])
    if units and num:
        if len(units) > 1 and re.sub(" ","",units[0]) == "":
                units = units[1].split()
        else:
            units = units[0].split()
        tmp = 0
        for n in num:
            tmp += float(n.replace(",","").replace(" ", ""))
        num = tmp / len(num)
        mult, unit = '', ''
        if units[0] in ['thousand', 'million', 'billion']:
            mult = units[0].lower()
            unit = ' '.join(units[1:]).lower()
        else:
            unit = ' '.join(units).lower()
        if mult not in ['thousand', 'million', 'billion'] and unit not in ['metric tons', 'kg', 'tons', 't']:
            num = np.nan
        else:
            if mult == 'thousand':
                num *= 1000
            if mult == 'million':
                num *= 1000000
            if mult == 'billion':
                num *= 1000000000
            if unit == 'kg':
                num *= 0.001
    else:
        num = np.nan
    row['Total Manufacturing Waste Generation'] = num

    # Waste Recycling and Diversion Rate % -> ratio
    tmp = re.sub(" ", "", re.sub("\\\\","",re.sub("[$]","",row['Waste Recycling and Diversion Rate'])))
    if '%' in tmp:
        num = np.mean(np.array(re.findall(r"([\d.]+)%", tmp)).astype('float'))/100.0
    else:
        num = np.nan
    if num >= 1.5:
        num = np.nan
    row['Waste Recycling and Diversion Rate'] = np.min([num,1.00]) 
    
    # Battery Recycling Rate % -> ratio
    tmp = re.sub(" ", "", re.sub("\\\\","",re.sub("[$]","",row['Battery Recycling Rate'])))
    if '%' in tmp:
        num = np.mean(np.array(re.findall(r"([\d.]+)%", tmp)).astype('float'))/100.0
    else:
        num = np.nan
    if num >= 1.5:
        num = np.nan
    row['Battery Recycling Rate'] = np.min([num,1.00]) 

    # Employee Count
    units = re.findall(r'\d[\d,\s.]+([A-Za-z\s³]+)', row['Employee Count'])
    num = re.findall(r'(\d[\d,\s.]+)[A-Za-z\s³]+', row['Employee Count'])
    if units and num:
        if len(units) > 1 and re.sub(" ","",units[0]) == "":
                units = units[1].split()
        else:
            units = units[0].split()
        tmp = 0
        for n in num:
            tmp += float(n.replace(",","").replace(" ", ""))
        num = tmp / len(num)
        mult, unit = '', ''
        if units[0] in ['thousand', 'million', 'billion']:
            mult = units[0].lower()
            unit = ' '.join(units[1:]).lower()
        else:
            unit = ' '.join(units).lower()
        if mult not in ['thousand', 'million', 'billion'] and unit not in ['count']:
            num = np.nan
        else:
            if mult == 'thousand':
                num *= 1000
            if mult == 'million':
                num *= 1000000
            if mult == 'billion':
                num *= 1000000000
    else:
        num = np.nan
    if num <= 0:
        num = np.nan
    row['Employee Count'] = [int(num) if not pd.isna(num) else num][0]

    # Employee Turnover Rate % -> ratio
    tmp = re.sub(" ","",re.sub("\\\\","",re.sub("[$]","",row['Employee Turnover Rate'])))
    if '%' in tmp:
        num = np.mean(np.array(re.findall(r"([\d.]+)%", tmp)).astype('float'))/100.0
    else:
        num = np.nan
    if num >= 1.5:
        num = np.nan
    row['Employee Turnover Rate'] = np.min([num,1.00]) 

    # Number of Workplace Accidents
    units = re.findall(r'\d[\d,\s.]+([A-Za-z\s³]+)', row['Number of Workplace Accidents'])
    num = re.findall(r'(\d[\d,\s.]+)[A-Za-z\s³]+', row['Number of Workplace Accidents'])
    if units and num:
        if len(units) > 1 and re.sub(" ","",units[0]) == "":
                units = units[1].split()
        else:
            units = units[0].split()
        tmp = 0
        for n in num:
            tmp += float(n.replace(",","").replace(" ", ""))
        num = tmp / len(num)
        mult, unit = '', ''
        if units[0] in ['thousand', 'million', 'billion']:
            mult = units[0].lower()
            unit = ' '.join(units[1:]).lower()
        else:
            unit = ' '.join(units).lower()
        if mult not in ['thousand', 'million', 'billion'] and unit not in ['count']:
            num = np.nan
        else:
            if mult == 'thousand':
                num *= 1000
            if mult == 'million':
                num *= 1000000
            if mult == 'billion':
                num *= 1000000000
    else:
        num = np.nan
    if num <= 0:
        num = np.nan
    row['Number of Workplace Accidents'] = [int(num) if not pd.isna(num) else num][0]

    # Employee Injury Rate
    tmp = re.sub(" ","",re.sub("\\\\","",re.sub("[$]","",row['Employee Injury Rate'])))
    if '%' in tmp:
        num = np.mean(np.array(re.findall(r"([\d.]+)%", tmp)).astype('float'))
    elif bool(re.fullmatch(r"\d+", tmp)):
        num = np.mean(np.array(re.findall(r"([\d.]+)", tmp)).astype('float'))
    else:
        num = np.nan
    row['Employee Injury Rate'] = num

    # Average Training Hours/Employee
    units = re.findall(r'\d[\d,\s.]+([A-Za-z\s³]+)', row['Average Training Hours/Employee'])
    num = re.findall(r'(\d[\d,\s.]+)[A-Za-z\s³]+', row['Average Training Hours/Employee'])
    if units and num:
        if len(units) > 1 and re.sub(" ","",units[0]) == "":
                units = units[1].split()
        else:
            units = units[0].split()
        tmp = 0
        for n in num:
            tmp += float(n.replace(",","").replace(" ", ""))
        num = tmp / len(num)
        mult, unit = '', ''
        if units[0] in ['thousand', 'million', 'billion']:
            mult = units[0].lower()
            unit = ' '.join(units[1:]).lower()
        else:
            unit = ' '.join(units).lower()
        if mult not in ['thousand', 'million', 'billion'] and unit not in ['hours']:
            num = np.nan
        else:
            if mult == 'thousand':
                num *= 1000
            if mult == 'million':
                num *= 1000000
            if mult == 'billion':
                num *= 1000000000
    else:
        num = np.nan
    if num <= 0:
        num = np.nan
    row['Average Training Hours/Employee'] = num

    # Training Investment/Employee
    units = re.findall(r'\d[\d,\s.]+([A-Za-z\s³()]+)', row['Training Investment/Employee'])
    num = re.findall(r'(\d[\d,\s.]+)[A-Za-z\s³()]+', row['Training Investment/Employee'])
    if units and num:
        if len(units) > 1 and re.sub(" ","",units[0]) == "":
                units = units[1].split()
        else:
            units = units[0].split()
        tmp = 0
        for n in num:
            tmp += float(n.replace(",","").replace(" ", "").replace("$","").replace("\\\\",""))
        num = tmp / len(num)
        mult, unit = '', ''
        if units[0] in ['thousand', 'million', 'billion']:
            mult = units[0].lower()
            unit = ' '.join(units[1:]).lower()
        else:
            unit = ' '.join(units).lower()
        if mult not in ['thousand', 'million', 'billion'] and unit not in ['currency (usd)']:
            num = np.nan
        else:
            if mult == 'thousand':
                num *= 1000
            if mult == 'million':
                num *= 1000000
            if mult == 'billion':
                num *= 1000000000
    else:
        num = np.nan
    if num <= 0:
        num = np.nan
    row['Training Investment/Employee'] = num

    # Workforce Gender Ratios % -> ratio
    tmp = re.sub(" ","",re.sub("\\\\","",re.sub("[$]","",row['Workforce Gender Ratios'])))
    if '%' in tmp:
        num = np.mean(np.array(re.findall(r"([\d.]+)%", tmp)).astype('float'))/100.0
    elif bool(re.fullmatch(r"\d+", tmp)):
        num = np.mean(np.array(re.findall(r"([\d.]+)", tmp)).astype('float'))/100.0
    else:
        num = np.nan
    if num >= 1.5:
        num = np.nan
    row['Workforce Gender Ratios'] = np.min([num,1.00]) 

    # Workforce Minority Ratios % -> ratio
    tmp = re.sub(" ","",re.sub("\\\\","",re.sub("[$]","",row['Workforce Minority Ratios'])))
    if '%' in tmp:
        num = np.mean(np.array(re.findall(r"([\d.]+)%", tmp)).astype('float'))/100.0
    elif bool(re.fullmatch(r"\d+", tmp)):
        num = np.mean(np.array(re.findall(r"([\d.]+)", tmp)).astype('float'))/100.0
    else:
        num = np.nan
    if num >= 1.5:
        num = np.nan
    row['Workforce Minority Ratios'] = np.min([num,1.00]) 
    
    # Number of Corruption Incidents
    units = re.findall(r'\d[\d,\s.]+([A-Za-z\s³]+)', row['Number of Corruption Incidents'])
    num = re.findall(r'(\d[\d,\s.]+)[A-Za-z\s³]+', row['Number of Corruption Incidents'])
    if units and num:
        if len(units) > 1 and re.sub(" ","",units[0]) == "":
                units = units[1].split()
        else:
            units = units[0].split()
        tmp = 0
        for n in num:
            tmp += float(n.replace(",","").replace(" ", ""))
        num = tmp / len(num)
        mult, unit = '', ''
        if units[0] in ['thousand', 'million', 'billion']:
            mult = units[0].lower()
            unit = ' '.join(units[1:]).lower()
        else:
            unit = ' '.join(units).lower()
        if mult not in ['thousand', 'million', 'billion'] and unit not in ['count']:
            num = np.nan
        else:
            if mult == 'thousand':
                num *= 1000
            if mult == 'million':
                num *= 1000000
            if mult == 'billion':
                num *= 1000000000
    else:
        num = np.nan
    if num <= 0:
        num = np.nan
    row['Number of Corruption Incidents'] = [int(num) if not pd.isna(num) else num][0]
    
    # Anti-corruption Compliance Rate % -> ratio
    tmp = re.sub(" ","",re.sub("\\\\","",re.sub("[$]","",row['Anti-corruption Compliance Rate'])))
    if '%' in tmp:
        num = np.mean(np.array(re.findall(r"([\d.]+)%", tmp)).astype('float'))/100.0
    elif bool(re.fullmatch(r"\d+", tmp)):
        num = np.mean(np.array(re.findall(r"([\d.]+)", tmp)).astype('float'))/100.0
    else:
        num = np.nan
    if num >= 1.5:
        num = np.nan
    row['Anti-corruption Compliance Rate'] = np.min([num,1.00]) 

    # Number of Anti-competitive Practices
    units = re.findall(r'\d[\d,\s.]+([A-Za-z\s³]+)', row['Number of Anti-competitive Practices'])
    num = re.findall(r'(\d[\d,\s.]+)[A-Za-z\s³]+', row['Number of Anti-competitive Practices'])
    if units and num:
        if len(units) > 1 and re.sub(" ","",units[0]) == "":
                units = units[1].split()
        else:
            units = units[0].split()
        tmp = 0
        for n in num:
            tmp += float(n.replace(",","").replace(" ", ""))
        num = tmp / len(num)
        mult, unit = '', ''
        if units[0] in ['thousand', 'million', 'billion']:
            mult = units[0].lower()
            unit = ' '.join(units[1:]).lower()
        else:
            unit = ' '.join(units).lower()
        if mult not in ['thousand', 'million', 'billion'] and unit not in ['count']:
            num = np.nan
        else:
            if mult == 'thousand':
                num *= 1000
            if mult == 'million':
                num *= 1000000
            if mult == 'billion':
                num *= 1000000000
    else:
        num = np.nan
    if num <= 0:
        num = np.nan
    row['Number of Anti-competitive Practices'] = [int(num) if not pd.isna(num) else num][0]

    # Training Investment/Employee
    units = re.findall(r'\d[\d,\s.]+([A-Za-z\s³()]+)', row['Monetary Value of Fines Imposed'])
    num = re.findall(r'(\d[\d,\s.]+)[A-Za-z\s³()]+', row['Monetary Value of Fines Imposed'])
    if units and num:
        if len(units) > 1 and re.sub(" ","",units[0]) == "":
                units = units[1].split()
        else:
            units = units[0].split()
        tmp = 0
        for n in num:
            tmp += float(n.replace(",","").replace(" ", "").replace("$","").replace("\\\\",""))
        num = tmp / len(num)
        mult, unit = '', ''
        if units[0] in ['thousand', 'million', 'billion']:
            mult = units[0].lower()
            unit = ' '.join(units[1:]).lower()
        else:
            unit = ' '.join(units).lower()
        if mult not in ['thousand', 'million', 'billion'] and unit not in ['currency (usd)']:
            num = np.nan
        else:
            if mult == 'thousand':
                num *= 1000
            if mult == 'million':
                num *= 1000000
            if mult == 'billion':
                num *= 1000000000
    else:
        num = np.nan
    if num <= 0:
        num = np.nan
    row['Monetary Value of Fines Imposed'] = num

    # Political Contributions and Lobbying Expenditures
    units = re.findall(r'\d[\d,\s.]+([A-Za-z\s³()]+)', row['Political Contributions and Lobbying Expenditures'])
    num = re.findall(r'(\d[\d,\s.]+)[A-Za-z\s³()]+', row['Political Contributions and Lobbying Expenditures'])
    if units and num:
        if len(units) > 1 and re.sub(" ","",units[0]) == "":
                units = units[1].split()
        else:
            units = units[0].split()
        tmp = 0
        for n in num:
            tmp += float(n.replace(",","").replace(" ", "").replace("$","").replace("\\\\",""))
        num = tmp / len(num)
        mult, unit = '', ''
        if units[0] in ['thousand', 'million', 'billion']:
            mult = units[0].lower()
            unit = ' '.join(units[1:]).lower()
        else:
            unit = ' '.join(units).lower()
        if mult not in ['thousand', 'million', 'billion'] and unit not in ['currency (usd)']:
            num = np.nan
        else:
            if mult == 'thousand':
                num *= 1000
            if mult == 'million':
                num *= 1000000
            if mult == 'billion':
                num *= 1000000000
    else:
        num = np.nan
    if num <= 0:
        num = np.nan
    row['Political Contributions and Lobbying Expenditures'] = num

    # Number of Marketing Compliance and Ethical Advertising Violation Incidents
    units = re.findall(r'\d[\d,\s.]+([A-Za-z\s³]+)', row['Number of Marketing Compliance and Ethical Advertising Violation Incidents'])
    num = re.findall(r'(\d[\d,\s.]+)[A-Za-z\s³]+', row['Number of Marketing Compliance and Ethical Advertising Violation Incidents'])
    if units and num:
        if len(units) > 1 and re.sub(" ","",units[0]) == "":
                units = units[1].split()
        else:
            units = units[0].split()
        tmp = 0
        for n in num:
            tmp += float(n.replace(",","").replace(" ", ""))
        num = tmp / len(num)
        mult, unit = '', ''
        if units[0] in ['thousand', 'million', 'billion']:
            mult = units[0].lower()
            unit = ' '.join(units[1:]).lower()
        else:
            unit = ' '.join(units).lower()
        if mult not in ['thousand', 'million', 'billion'] and unit not in ['count']:
            num = np.nan
        else:
            if mult == 'thousand':
                num *= 1000
            if mult == 'million':
                num *= 1000000
            if mult == 'billion':
                num *= 1000000000
    else:
        num = np.nan
    if num <= 0:
        num = np.nan
    row['Number of Marketing Compliance and Ethical Advertising Violation Incidents'] = [int(num) if not pd.isna(num) else num][0]

    # Sales-weighted Average Fuel Economy/Emissions
    units = re.findall(r'\d[\d,\s.]+([A-Za-z\s³]+)', row['Sales-weighted Average Fuel Economy/Emissions'])
    num = re.findall(r'(\d[\d,\s.]+)[A-Za-z\s³]+', row['Sales-weighted Average Fuel Economy/Emissions'])
    if units and num:
        if len(units) > 1 and re.sub(" ","",units[0]) == "":
                units = units[1].split()
        else:
            units = units[0].split()
        tmp = 0
        for n in num:
            tmp += float(n.replace(",","").replace(" ", ""))
        num = tmp / len(num)
        mult, unit = '', ''
        if units[0] in ['thousand', 'million', 'billion']:
            mult = units[0].lower()
            unit = ' '.join(units[1:]).lower()
        else:
            unit = ' '.join(units).lower()
        if mult not in ['thousand', 'million', 'billion'] and unit not in ['mpg', 'gco']:
            num = np.nan
        else:
            if mult == 'thousand':
                num *= 1000
            if mult == 'million':
                num *= 1000000
            if mult == 'billion':
                num *= 1000000000
            if unit == 'gco':
                num = 8887/(num*1.609)
    else:
        num = np.nan
    row['Sales-weighted Average Fuel Economy/Emissions'] = num

    # Zero Emission and Alternative Fuel Vehicle Sales
    units = re.findall(r'\d[\d,\s.]+([A-Za-z\s³]+)', row['Zero Emission and Alternative Fuel Vehicle Sales'])
    num = re.findall(r'(\d[\d,\s.]+)[A-Za-z\s³]+', row['Zero Emission and Alternative Fuel Vehicle Sales'])
    if units and num:
        if len(units) > 1 and re.sub(" ","",units[0]) == "":
                units = units[1].split()
        else:
            units = units[0].split()
        tmp = 0
        for n in num:
            tmp += float(n.replace(",","").replace(" ", ""))
        num = tmp / len(num)
        mult, unit = '', ''
        if units[0] in ['thousand', 'million', 'billion']:
            mult = units[0].lower()
            unit = ' '.join(units[1:]).lower()
        else:
            unit = ' '.join(units).lower()
        if mult not in ['thousand', 'million', 'billion'] and unit not in ['vehicle units']:
            num = np.nan
        else:
            if mult == 'thousand':
                num *= 1000
            if mult == 'million':
                num *= 1000000
            if mult == 'billion':
                num *= 1000000000
    else:
        num = np.nan
    if num <= 0:
        num = np.nan
    row['Zero Emission and Alternative Fuel Vehicle Sales'] = [int(num) if not pd.isna(num) else num][0]

    return row

def group_company(row):
    if row['size'] in ['10K+', '5K-10K']:
        row['size'] = 'Large'
    elif row['size'] in ['501-1K', '1K-5K']:
        row['size'] = 'Medium'
    else:
        row['size'] = 'Small'
    return row

def fit_minmax(row):
    for score in ['esg_score', 'e_score', 's_score', 'g_score']:
        if row[score] > 100:
            row[score] = 100
        if row[score] < 0:
            row[score] = 0
        row[score] = np.round(row[score], 2)
    return row

def parse_filename(filename):
    try:
        # Remove path and extension
        clean_name = os.path.splitext(os.path.basename(filename))[0]
        
        # Extract year using regex
        year_match = re.search(r'-(\d{4})(\s|$)', clean_name)
        year = int(year_match.group(1)) if year_match else None
        
        # Remove year and split remaining parts
        base = re.sub(r'-\d{4}.*', '', clean_name)
        parts = base.split('-')
        
        # Handle different industry-company patterns
        if len(parts) >= 2:
            if 'Major' in parts[1]:
                industry = '-'.join(parts[0:2]).strip()
                company = '-'.join(parts[2:]).strip()
            else:
                industry = parts[0].strip()
                company = '-'.join(parts[1:]).strip()
        else:
            industry = base.strip()
            company = 'Not Reported'
        
        return pd.Series([industry, company, year])
    
    except Exception as e:
        print(f"Error parsing {filename}: {str(e)}")
        return pd.Series([None, None, None])

def get_size(count):
    sizes = []
    for cnt in count:
        size = '1K-5K'
        if cnt:
            if cnt <= 500:
                size = '201-500'
            elif cnt <= 1000:
                size = '501-1K'
            elif cnt <= 5000:
                size = '1K-5K'
            elif cnt <= 10000:
                size = '5K-10K'
            else:
                size = '10K+'
        sizes.append(size)
    return sizes