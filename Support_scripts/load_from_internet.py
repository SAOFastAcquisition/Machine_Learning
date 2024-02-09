# url = '//codeload.github.com/fogleman/Minecraft/zip/master'
url = 'https://solar.sao.ru/data/fast-acquisition/1-3ghz/2024/01/'
# downloading with requests
from html.parser import HTMLParser
# import the requests library
import requests
from html.parser import HTMLParser
from html.entities import name2codepoint

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start tag:", tag)
        for attr in attrs:
            print("     attr:", attr)

    def handle_endtag(self, tag):
        print("End tag  :", tag)

    def handle_data(self, data):
        print("Data     :", data)

    def handle_comment(self, data):
        print("Comment  :", data)

    def handle_entityref(self, name):
        c = chr(name2codepoint[name])
        print("Named ent:", c)

    def handle_charref(self, name):
        if name.startswith('x'):
            c = chr(int(name[1:], 16))
        else:
            c = chr(int(name))
        print("Num ent  :", c)

    def handle_decl(self, data):
        print("Decl     :", data)


# создаем объект парсера
parser = MyHTMLParser()
# download the file contents in binary format
r = requests.get(url)
h = requests.head(url).text
cont = r.content
a = r.raw
for chunk in r:
    parser.handle_data(chunk)



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