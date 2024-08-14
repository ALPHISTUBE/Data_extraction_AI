# serializers.py
from rest_framework import serializers

class ImageSerializer(serializers.Serializer):
    image_data = serializers.CharField()