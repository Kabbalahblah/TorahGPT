import json
import os
import re
import urllib.request

URL = "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json"
FILENAME = "en_kjv.json"

if not os.path.exists(FILENAME):           # download once
    urllib.request.urlretrieve(URL, FILENAME)

with open(FILENAME, encoding="utf-8") as f:
    books = json.load(f)

# typographic characters -> plain keyboard equivalents, so typed prompts use the same characters
REPLACE = {"’": "'", "—": "-", "æ": "ae", "Æ": "Ae"}

verses = []
for book in books:
    for chapter in book["chapters"]:
        for verse in chapter:
            if "…" in verse:
                # a translator's marginal note leaked into 8 verses: cut back to the last full sentence
                verse = verse[:verse.index("…")]
                verse = verse[:max(verse.rfind("."), verse.rfind("?"), verse.rfind("!")) + 1]
            for old, new in REPLACE.items():
                verse = verse.replace(old, new)
            verse = re.sub(r" +([,.;:?!])", r"\1", verse)    # "the LORD ," -> "the LORD,"
            verse = re.sub(r"\( +", "(", verse)                # "( For" -> "(For"
            verse = re.sub(r" +\)", ")", verse)                # "LORD )" -> "LORD)"
            # leaked notes start with punctuation genuine KJV never uses, e.g. "day., because" or "LORD:: or,"
            verse = re.sub(r"([.?:])[.,:].*$", r"\1", verse)
            # or with a lowercase fragment after the final full stop, e.g. "the Abi-ezrites. send peace"
            if verse[-1:] not in ".,;:?!)'":
                verse = re.sub(r"([.?]) [a-z][^.?]*$", r"\1", verse)
            verses.append(verse.strip())

text = "\n".join(verses) + "\n"            # one verse per line
with open("kjv.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Verses:", len(verses))
print("Characters:", len(text))
print("Distinct characters:", len(set(text)))
print(verses[0])