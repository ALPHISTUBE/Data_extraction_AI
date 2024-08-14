import cv2
import glob
import pytesseract
from spellchecker import SpellChecker
import re
import numpy as np
from dateutil import parser
from PIL import Image

from django.shortcuts import render
from .forms import UploadImageForm
from django.core.files.storage import FileSystemStorage

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def index(request):
    extracted_text = ""
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            imagef = form.cleaned_data['image']
            fs = FileSystemStorage()
            filename = fs.save(f"{imagef.name}.jpg", imagef)
            uploaded_file_url = fs.url(filename)

            # Extract text and get image URLs
            extracted_text, dpi = getText2CV(filename)
            print(extracted_text)
            # Process extracted text
            currectionData, tm, date = formatTextFromReceipt(extracted_text)
            print(currectionData)
            print(date)

            return render(request, 'iocr/index.html', {
                'form': form,
                'uploaded_file_url': uploaded_file_url,
                'extracted_text': currectionData,
                'taxCount': tm,
                'date' : date,
                "dpi" : dpi
            })
    else:
        form = UploadImageForm()
    return render(request, 'iocr/index.html', {'form': form, 'extracted_text': extracted_text})

#Image to text

#version 2

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

#version 1
spell = SpellChecker()
def currectSpelling(text):
    words = text.split()
    corrected_words = ""
    
    for word in words:
        # Check if the word is misspelled
        if word not in spell:
            # Correct the word
            corrected_word = spell.correction(word)
            corrected_words += f"{corrected_word} "
        else:
           corrected_words += f"{word} "
    
    return corrected_words

def formatTextFromReceipt(text : str):
    et = {}
    taxMax = 0

    lines = text.splitlines(text)
    dates = []
    indexl = 0
    for line in lines:
        indexl += 1
        line = line.lower()
        #line = line.replace("~","-")
        #line = line.replace("\n","")
        #line = currectSpelling(line)
        et[f"Line:{indexl}"] = line
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
