import os
import urllib.request
import xml.etree.ElementTree as ET

BOOKS = ["Gen", "Exod", "Lev", "Num", "Deut"]
URL = "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/{}.xml"
NS = "{http://www.bibletechnologies.net/2003/OSIS/namespace}"

def letters_only(s):
    # keep only the 27 Hebrew letter forms, U+05D0 (alef) to U+05EA (tav)
    return "".join(ch for ch in s if "\u05d0" <= ch <= "\u05ea")

verses = []
for book in BOOKS:
    filename = f"{book}.xml"
    if not os.path.exists(filename):              # download each book once
        urllib.request.urlretrieve(URL.format(book), filename)
    root = ET.parse(filename).getroot()
    for verse in root.iter(NS + "verse"):
        # findall only looks at direct children, so words inside notes are skipped
        words = [letters_only("".join(w.itertext())) for w in verse.findall(NS + "w")]
        verses.append(" ".join(words))

text = "\n".join(verses) + "\n"                   # one verse per line
with open("torah.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Verses:", len(verses))
print("Words:", sum(len(v.split()) for v in verses))
print("Letters:", sum(len(letters_only(v)) for v in verses))
print("Characters in file:", len(text))
print("Distinct characters:", len(set(text)))
print(verses[0])