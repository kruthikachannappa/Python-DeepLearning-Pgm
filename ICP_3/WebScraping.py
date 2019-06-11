from bs4 import BeautifulSoup
import requests

url = "https://en.wikipedia.org/wiki/Deep_learning"
response = requests.get(url)
soup = BeautifulSoup(response.text,'html.parser')
result = soup.find_all('p',{'class':"reference"})
for res in result:
    print(res)
print("******************TITLE**************************")
print(soup.title.string)
print("******************ALL <a></a> Tags**************************")
print(soup.findAll('a'))
all_links = soup.find_all("a")
print("**********************ALL hrefs****************************")
for link in all_links:
    print(link.get("href"))
