import cv2
import glob
import pytesseract
import re
import numpy as np
from dateutil import parser
from PIL import Image

from django.core.files.storage import FileSystemStorage

def Scan(imagef):
    fs = FileSystemStorage()
    filename = fs.save(imagef.name, imagef)
    # Extract text and get image URLs
    extracted_text, dpi = getText2CV(filename)
    print(extracted_text)
    # Process extracted text
    currectionData, tm, date = formatTextFromReceipt(extracted_text)

    data = [
        "recepit",
        dpi,
        date,
        tm,
        currectionData
    ]
    print(data)

    return data


#Image Processing
def autoScale(img, path):
    dpi = findImgDPI(path)
    if dpi != 1:
        targetedDpi = 300
        scaleMultiplier = float(targetedDpi / dpi)
    else:
        scaleMultiplier = 1.2
    img = cv2.resize(img, None, fx=scaleMultiplier, fy=scaleMultiplier, interpolation=cv2.INTER_CUBIC)
    return img

def processImg(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def findImgDPI(path):

    try:
        with Image.open(path) as img:
            dpi = img.info.get("dpi")

            if dpi:
                return dpi[0]
            else:
                return 1.0
    except:
        pass

def getText2CV(filename):
    fs = FileSystemStorage()
    img_path = fs.path(filename)

    img = cv2.imread(img_path)

    img = processImg(autoScale(img, img_path))

    dpi = findImgDPI(img_path)
    
    return pytesseract.image_to_string(img), dpi

#Text Formating
receiptKey = [
    'total',
    'cash',
    'change',
    'subtotal',
    'vat',
    'tax',
    'totaltax',
    'discount',
]

bankKey = [
    'debit',
    'credit',
    'visa',
    'mcard',
    'check'
]

receiptKey += bankKey

requiretRecepitField = {
    'total' : [],
    'subtotal': [],
    'change' : [],
    'cash' : [],
}

#version 1

def formatTextFromReceipt(text : str):
    et = {}
    taxMax = 0
    bankT = False

    lines = text.splitlines(text)
    dates = []
    for line in lines:
        line = line.lower()
        
        #Date
        try:
            # Try parsing the date
            #date = parser.parse(line, fuzzy=True)
            date = extract_dates(line)
            if len(date) != 0:
                dates.append(date[0])
        except ValueError:
            # Not a date
            continue
        
        #Data
        for kw in receiptKey:
            
            if line.find(kw) != -1:
                stpos = line.find('tax')
                
                if stpos != -1 and stpos != 0:
                    text = line[0:stpos]
                    t = text.find('total')
                    if t != -1 and t != 0:
                        text2 = line[0:t]
                        line = line.removeprefix(text2)
                        # print(f"{kw}:{t},{text},{line}")
                    else:
                        line = line.removeprefix(text)
                        # print(f"{kw}:{t},{text},{line}")

                stpos = line.find('total')
                
                if stpos != -1 and stpos != 0:
                    t = text.find('sub')
                    if t != -1 and t != 0:
                        text2 = line[0:t]
                        line = line.removeprefix(text2)
                        # print(f"{kw}:{t},{text},{line}")
                    else:
                        line = line.removeprefix(text)
                        # print(f"{kw}:{t},{text},{line}")

                stpos = line.find(kw)
                
                if line.find('total') == -1 and line.find('tax') == -1 and stpos != -1 and stpos != 0:
                    text = line[0:stpos]
                    line = line.removeprefix(text)
                
                    # print(f"{kw}:{text},{line}")

            if et.get(kw) == None and line.startswith(kw) and kw != 'tax':
                line = line.replace(" ", "")
                line = line.replace(",", ".")
                if kw == "visa" or kw == "debit" or kw == "credit":
                    bankT = True
                data = re.findall(r"\d+\.\d+", line)
                float_data = [float(num) for num in data]

                if len(float_data) != 0:
                    et[kw] = float_data[0]

            if(line.startswith(kw) and kw == "tax" and line.find('total') == -1): 
                line = line.replace(",", ".")
                taxKw = f'{taxMax}.tax'
                data = re.findall(r"\d+\.\d+", line)
                float_data = [float(num) for num in data]

                if type(float_data) == list and len(float_data) > 1:
                    float_data.pop(0)

                if len(float_data) != 0:
                    et[taxKw] = float_data[0]
                    requiretRecepitField[taxKw] = {}
                    taxMax += 1

    noTaxText = False
    if taxMax == 0:
        noTaxText = True
        taxMax = 1
        requiretRecepitField["tax0"] = {}

    # print(dates)
    et = validate_and_fill_receipt_data(et, noTaxText)

    if len(dates) != 0:
        return et, taxMax, dates[0]
    else:
        return et, taxMax, "No Date Found"
def extract_dates(text):
    # Define regex pattern for common date formats (e.g., MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD)
    date_patterns = [
        r'\b\d{2}/\d{2}/\d{4}\b',  # MM/DD/YYYY
        r'\b\d{2}-\d{2}-\d{4}\b',  # MM-DD-YYYY
        r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
        r'\b\d{2}/\d{2}/\d{2}\b',  # MM/DD/YY
        r'\b\d{2}-\d{2}-\d{2}\b'   # DD-MM-YY
    ]
    
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text))
    
    return dates

def common_cash_change_validation_algo(total, cash, change, bank_transfer):
    # Calculate and validate cash and change if it's a cash transaction
    if not bank_transfer:
        if total is not None and cash is not None and change is not None:
            # Validate change
            if change != cash - total:
                change = cash - total

        if total is not None and change is not None and cash is not None:
            # Validate cash
            if cash != total + change:
                cash = total + change

def validate_and_fill_receipt_data(receipt_data, noTaxTextFoundInReceipt):
    # Original dictionary (for comparison)
    original_data = receipt_data.copy()
    
    # Initialize variables to store values from the receipt
    total = receipt_data.get('total')
    cash = receipt_data.get('cash')
    change = receipt_data.get('change')
    subtotal = receipt_data.get('subtotal')
    vat = receipt_data.get('vat')
    discount = receipt_data.get('discount')
    debit = receipt_data.get('debit')
    credit = receipt_data.get('credit')
    visa = receipt_data.get('visa')
    mcard = receipt_data.get('mcard')

    # Collect and sum all taxes
    taxes = {key: value for key, value in receipt_data.items() if key.find('.tax') != -1}
    total_tax = float(sum(taxes.values()))
    # print(taxes.values())

    # Check if bank transfer is made (debit, credit, or visa)
    bank_transfer = debit or credit or visa or mcard

    # Validate existing data
    if noTaxTextFoundInReceipt:
        # No tax means total should equal subtotal
        if total is not None and subtotal is not None:
            if total != subtotal:
                # Validate using other values
                if cash is not None and change is not None:
                    # Checking validation for cash and change
                    common_cash_change_validation_algo(total, cash, change, bank_transfer)

                    # If cash and change are provided, use them to validate total
                    val1 = cash - change
                    val2 = total + cash
                    val3 = total + change

                    if val1 == val2:
                        total = cash - change
                        subtotal = total
                    elif val1 == val3:
                        subtotal = cash - change
                        total = subtotal
                elif debit is not None:
                    total = debit
                    subtotal = total
                elif credit is not None:
                    total = credit
                    subtotal = total
                elif visa is not None:
                    total = visa
                    subtotal = total
                elif mcard is not None:
                    total = mcard
                    subtotal = total
                else:
                    subtotal = total
        elif total is not None and subtotal is None:
            subtotal = total
        elif subtotal is not None and total is None:
            total = subtotal
    else:
        if total is not None and subtotal is not None:
            val2 = total - subtotal
            if total_tax != val2:
                total_tax = val2

        if total is not None:
            if total_tax is not None and subtotal is not None:
                # Validate subtotal
                if subtotal != total - total_tax:
                    subtotal = total - total_tax

            if vat is not None and subtotal is not None:
                # Validate subtotal
                if subtotal != total - vat:
                    subtotal = total - vat

            if discount is not None and subtotal is not None:
                # Validate subtotal
                if subtotal != total + discount:
                    subtotal = total + discount

        if subtotal is not None:
            if total_tax is not None and total is not None:
                # Validate total
                if total != subtotal + total_tax:
                    total = subtotal + total_tax

            if vat is not None and total is not None:
                # Validate total
                if total != subtotal + vat:
                    total = subtotal + vat

            if discount is not None and total is not None:
                # Validate total
                if total != subtotal - discount:
                    total = subtotal - discount

    # Calculate and validate cash and change if it's a cash transaction
    common_cash_change_validation_algo(total, cash, change, bank_transfer)

    # Validate bank transactions
    if bank_transfer:
        if total is not None:
            if debit is not None and credit is None and visa is None:
                # Validate debit
                if debit != total:
                    debit = total
            elif credit is not None and debit is None and visa is None:
                # Validate credit
                if credit != total:
                    credit = total
            elif visa is not None and debit is None and credit is None:
                # Validate visa
                if visa != total:
                    visa = total

    # Fill missing data
    if total is not None:
        if noTaxTextFoundInReceipt:
            subtotal = total
        else:
            if total_tax is not None and subtotal is None:
                subtotal = total - total_tax

            if vat is not None and subtotal is None:
                subtotal = total - vat

            if discount is not None and subtotal is None:
                subtotal = total + discount

    if subtotal is not None:
        if noTaxTextFoundInReceipt:
            total = subtotal
        else:
            if total_tax is not None and total is None:
                total = subtotal + total_tax

            if vat is not None and total is None:
                total = subtotal + vat

            if discount is not None and total is None:
                total = subtotal - discount

    if not bank_transfer:
        if total is not None and cash is not None and change is None:
            change = cash - total

        if total is not None and change is not None and cash is None:
            cash = total + change

    # Create a new dictionary with filled and validated data
    validated_data = {
        'subtotal': subtotal,
        'total tax': total_tax,
        'vat': vat,
        'total': total,
        'cash': cash,
        'change': change,
        'discount': discount,
        'debit': debit,
        'credit': credit,
        'visa': visa
    }

    # Remove None values
    # if validated_data['total_tax'] == 0 and len(taxes) == 0:
    #     validated_data['total_tax'] = None
    validated_data = {k: v for k, v in validated_data.items() if v is not None}


    # Create a comparison dictionary
    comparison_data = {}
    for key in set(receipt_data.keys()).union(validated_data.keys()):
        if original_data.get(key) == None and validated_data.get(key) == None:
            comparison_data[key] = "Missing"
        elif key in original_data and key in validated_data and original_data[key] == validated_data[key]:
            comparison_data[key] = "Accurate"
        elif key not in original_data and key in validated_data:
            original_data[key] = 0.0
            comparison_data[key] = "Fixed"
        elif key in original_data and key not in validated_data:
            validated_data[key] = 0.0
            comparison_data[key] = "Not Sure"
        elif key in original_data or key in validated_data:
            comparison_data[key] = "Fixed"

    # print(original_data)
    # print(validated_data)
    # print(comparison_data)

    et = {}
    for kw in validated_data:
        dt1 = "%.2f" % validated_data[kw]
        if kw in original_data:
            dt = original_data[kw]
        else:
            dt = 0.0
        dt2 = comparison_data[kw]
        
        li = [dt, dt1, dt2]

        et[kw] = li

    for kw in receiptKey:
        if et.get(kw) == None:
            li = [0.0, 0.0, "Missing"]
            et[kw] = li

    return et