from bs4 import BeautifulSoup
import requests
import csv

url = "https://www.aliexpress.com/"
#source = requests.get(url).text
with open("test.html") as html:
    soup = BeautifulSoup(html, "lxml")

mainCategory = []
subCategory = []

for a in soup.find_all('div', class_="Categoey--categoryItemTitle--2uJUqT2"):
    mainCategory.append(a.text)
for a in soup.find_all('div', class_="Categoey--cateItemLv3Title--1mjlI-5"):
    subCategory.append(a.text)

with open('mainCategory.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Main Category"])
    for entity in subCategory:
        writer.writerow([entity])