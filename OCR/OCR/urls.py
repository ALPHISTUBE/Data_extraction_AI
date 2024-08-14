from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('Home.urls')),  # Ensure 'Home.urls' is your app's URLs module
    path('', include('InvoiceOCR.urls')),  # Ensure 'Home.urls' is your app's URLs module
    path('', include('Api.urls')),  # Ensure 'Home.urls' is your app's URLs module
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
