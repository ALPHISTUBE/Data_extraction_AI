import cv2
import pytesseract
from spellchecker import SpellChecker
import re
import numpy as np
from dateutil import parser
from PIL import Image
import os
from django.shortcuts import render
from .forms import UploadImageForm
from django.core.files.storage import FileSystemStorage

def index(request):
    extracted_text = ""
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            imagef = form.cleaned_data['image']

            # Open the image using Pillow
            img = Image.open(imagef)

            # If the image is in RGBA mode, convert it to RGB
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            # Define the filename and file path for saving
            fs = FileSystemStorage()
            filename = f"{os.path.splitext(imagef.name)[0]}_fixed_dpi.jpg"
            filepath = os.path.join(fs.location, filename)

            # Set DPI and save the image directly to the file path
            img.save(filepath, format='JPEG', dpi=(150, 150))

            # Get the URL of the saved file
            uploaded_file_url = fs.url(filename)

            # Extract text and get image URLs
            extracted_text, actual_dpi = getText2CV(filepath)
            print(extracted_text)

            # Process extracted text
            currectionData, date, totalA = formatTableTextFromLine(extracted_text)

            return render(request, 'iocr/index.html', {
                'form': form,
                'uploaded_file_url': uploaded_file_url,
                'extracted_text': currectionData,
                'totalA': totalA,
                'date': date,
                "dpi": actual_dpi
            })
    else:
        form = UploadImageForm()
    
    return render(request, 'iocr/index.html', {'form': form, 'extracted_text': extracted_text})

def autoScale(img, path):
    dpi = findImgDPI(path)
    print(dpi)
    targetedDpi = 300
    scaleMultiplier = float(targetedDpi / dpi) if dpi != 1 else 1.2
    img = cv2.resize(img, None, fx=scaleMultiplier, fy=scaleMultiplier, interpolation=cv2.INTER_CUBIC)
    return img

def processImg(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def findImgDPI(path):
    try:
        with Image.open(path) as img:
            dpi = img.info.get('dpi')
            return dpi[0] if dpi else 1
    except Exception as e:
        print(f"Error finding DPI: {e}")
        return None

def getText2CV(filename):
    fs = FileSystemStorage()
    img_path = fs.path(filename)

    img = cv2.imread(img_path)

    img = processImg(autoScale(img, img_path))

    dpi = findImgDPI(img_path)
    
    return pytesseract.image_to_string(img), dpi

spell = SpellChecker()
def currectSpelling(text):
    words = text.split()
    corrected_words = " ".join([spell.correction(word) if word not in spell else word for word in words])
    return corrected_words

def formatTableTextFromLine(text):
    et = {}
    lines = text.splitlines()
    dates = []
    mainDate = ""
    total_amount = 0.0
    indexl = 0
    for line in lines:
        line = line.lower().replace("\n", "").replace("_", " ")
        date = ""
        dt_ext = extract_dates(line)
        if len(dt_ext) > 0:
            indexl += 1
            date = dt_ext[0]
            dates.append(date)
            line = line.replace(date, "").replace(",", "")
            data = re.findall(r"\d+\.\d+", line)
            float_data = [float(num) for num in data] if data else []
            if float_data:
                total_amount += float_data[0]
                line = line.replace(data[0], "")
                et[f"{indexl}"] = [date, line, data[0]]
            else:
                et[f"{indexl}"] = [date, line, "NONE"]

    mainDate = dates[0]
    algoWorked = CheckDateAndAmountRadio(et)
    print(algoWorked)
    total_amount = round(total_amount, 2)
    if not algoWorked:
        et, mainDate, total_amount = formatTextFrom3ColumeTable(text)

    if mainDate != "":
        return et, mainDate, total_amount
    else:
        return et, "No Date Found", total_amount


tableHeaderKey = [
    "date",
    "description",
    "amount"
]

usa_st_code = [
    'AK', 'AL', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE',
    'FL', 'FM', 'GA', 'GU', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS',
    'KY', 'LA', 'MA', 'MD', 'ME', 'MH', 'MI', 'MN', 'MO', 'MP',
    'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY',
    'OH', 'OK', 'OR', 'PA', 'PR', 'PW', 'RI', 'SC', 'SD', 'TN',
    'TX', 'UT', 'VA', 'VI', 'VT', 'WA', 'WI', 'WV', 'WY', "WEB"
]

def formatTextFrom3ColumeTable(text):
    et = {}
    dates = []
    descriptions = []
    amounts = []
    lines = text.splitlines()
    total_amount = 0.0

    indexl = 0
    headerIndex = 0
    headerText = ""
    desI = 0
    amoI = 0
    for line in lines:
        line = str(line.lower())

        if headerIndex <= 2 and line.find(tableHeaderKey[headerIndex]) != -1:
            headerText = tableHeaderKey[headerIndex]
            headerIndex += 1
        
        
        #Date
        dt_ext = extract_dates(line)
        if headerText == "date" and line != "":
            if len(dt_ext) > 0:
                dates.append(dt_ext[0])            

        #Description
        if headerText == "description" and line != "":

            codes = line.split(" ")
            hasCode = codes[len(codes) - 1].upper() in usa_st_code
            print(hasCode)
            if len(dt_ext) == 0 and desI != 0 and hasCode:
                descriptions.append(line)
            if len(dt_ext) == 0 and desI != 0 and not hasCode:
                text = descriptions[len(descriptions) - 1]
                descriptions.append(f"{text}\n{line}")
            desI = 1

        #Amount
        if headerText == "amount" and line != "":
            line = line.replace(",", "")
            
            if amoI != 0:
                # data = re.findall(r"\d+\.\d+", line)
                data = getAmount(line)
                float_data = [float(num) for num in data] if data else []
                if float_data:
                    total_amount += float_data[0]
                    line = line.replace(data[0], "")
                    amounts.append(data[0])
                else:
                    amounts.append("NONE")
            amoI = 1

        dt_ext = extract_dates(line)

    print(len(descriptions))
    for date in dates:
        et[f"{indexl}"] = [date, descriptions[indexl], amounts[indexl]]
        indexl += 1

    total_amount = round(total_amount, 2)
    if dates[0] != "":
        return et, dates[0], total_amount
    else:
        return et, "No Date Found", total_amount

def CheckDateAndAmountRadio(formattedData : dict):
    
    totalTranc = len(formattedData)
    dataFound = 0
    
    for data in formattedData.values():
        print(data)
        if data[0] != "" and data[1] != "" and data[2] != "NONE":
            dataFound += 1
        else:
            dataFound -+ 1

    if dataFound <= totalTranc / 2:
        return False
    else:
        return True

def getAmount(line):
    print(line)
    data = re.findall(r"\d+\.\d+", line)
    data1 = re.findall(r"\d+\.\d+\.\d+", line)
    data2 = re.findall(r"\.\d+", line)

    if data:
        return data
    elif data1:
        text = data1[0][0:len(line) - 3]
        data = re.findall(r"\d+\.\d+", text)
        if data:
            return data
    elif data2:
        text = "0" + data2[0]
        data2[0] = text
        return data2
    else:
        return ["NONE"]

def extract_dates(text):
    date_patterns = [
        r'\b\d{2}/\d{2}/\d{4}\b',  
        r'\b\d{2}-\d{2}-\d{4}\b',  
        r'\b\d{4}-\d{2}-\d{2}\b',  
        r'\b\d{2}/\d{2}/\d{2}\b',  
        r'\b\d{2}-\d{2}-\d{2}\b',   
        r'\b\d{2}/\d{2}\b',  
        r'\b\d{2}-\d{2}\b',  
        r'\b\d{4}-\d{2}\b',  
    ]
    
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text))
    
    return dates
