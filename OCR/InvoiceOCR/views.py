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
            currectionData, date, totalA = formatTextFromReceipt(extracted_text)

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

def formatTextFromReceipt(text):
    et = {}
    lines = text.splitlines()
    dates = []
    total_amount = 0.0
    indexl = 0
    for line in lines:
        line = line.lower().replace("\n", "").replace("_", " ")
        date = ""
        dt_ext = extract_dates(line)

        if len(dt_ext) > 0:
            indexl += 1
            date = dt_ext[0]
            line = line.replace(date, "").replace(",", "")
            data = re.findall(r"\d+\.\d+", line)
            float_data = [float(num) for num in data] if data else []
            if float_data:
                total_amount += float_data[0]
                line = line.replace(data[0], "")
                et[f"{indexl}"] = [date, line, data[0]]
            else:
                et[f"{indexl}"] = [date, line, "NONE"]

    if dates:
        return et, dates[0], total_amount
    else:
        return et, "No Date Found", total_amount

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
