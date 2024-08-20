from django import forms

class UploadFileForm(forms.Form):
    book_file = forms.FileField(label='Select book pass CSV')
    bank_file = forms.FileField(label='Select bank pass CSV')
