from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializers import ImageSerializer
import cv2
import numpy as np
import pytesseract

import random

from .scanReceipt import Scan

@api_view(['POST'])
def process_image(request):
    # Check if an image file was uploaded
    if 'image' in request.FILES:
        image_file = request.FILES['image']

        data = Scan(image_file)
        
        return Response({'data': data}, status=200)
    
    return Response({'error': 'No image uploaded'}, status=400)
