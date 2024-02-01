# url = '//codeload.github.com/fogleman/Minecraft/zip/master'
url = 'https://solar.sao.ru/data/fast-acquisition/1-3ghz/2024/01/'
# downloading with requests

# import the requests library
import requests

# download the file contents in binary format
r = requests.get(url)
h = requests.head(url).text
cont = r.content
a = r.raw
# b = r.json()

# open method to open a file on your system and write the contents
with open("list.txt.gz", "wb") as code:
    code.write(r.content)

# downloading with urllib

# import the urllib library

# Copy a network object to a local file
import urllib.request
data, headers = urllib.request.urlretrieve(url)
pass