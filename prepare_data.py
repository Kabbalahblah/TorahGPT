import os
import urllib.request
import xml.etree.ElementTree as ET

BOOKS = ["Gen", "Exod", "Lev", "Num", "Deut"]
URL = "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/{}.xml"
NS = "{http://www.bibletechnologies.net/2003/OSIS/namespace}"

# Divine names are identified by their Strong's numbers (the "lemma" tags in the source)
# and written with the substitutes customary in Orthodox writing, so the model never
# sees the names themselves. Ordinary words spelled like names, such as "to" or
# "my lord" said to a person, carry different numbers and are left as they are.
SUBSTITUTES = {
    "3068": lambda s: "ה׳",                          # the Tetragrammaton (with a geresh)
    "3069": lambda s: "ה׳",                          # the Tetragrammaton, read as Elohim
    "430": lambda s: s.replace("ה", "ק", 1),         # Elohim and its forms: אלקים, אלקיך...
    "433": lambda s: s.replace("ה", "ק", 1),         # Eloah: אלוק
    "410": lambda s: "קל" if s == "אל" else s,       # El (the plural, meaning "gods", is left as it is)
    "136": lambda s: s.replace("אדנ", "אדנ-", 1),    # Adonai: אדנ-י
    "7706": lambda s: s.replace("שד", "שד-", 1),     # Shaddai: שד-י
    "3050": lambda s: "י-ה",                         # Yah
}
# Ehyeh in Exodus 3:14 is tagged as the ordinary verb "to be", so it's identified by verse
EHYEH = ("Exod.3.14", "1961", "HVqi1cs")
# The Tetragrammaton written as escape codes, so the name doesn't appear in this file
TETRAGRAMMATON = "\u05d9\u05d4\u05d5\u05d4"

def letters_only(s):
    # keep only the 27 Hebrew letter forms, U+05D0 (alef) to U+05EA (tav)
    return "".join(ch for ch in s if "\u05d0" <= ch <= "\u05ea")

def word_text(w, verse_id):
    # the word's text and its tags are both split into parts by "/" (prefixes, then the word)
    parts = [letters_only(p) for p in "".join(w.itertext()).split("/")]
    for i, tag in enumerate(w.get("lemma", "").split("/")):
        number = tag.strip().split(" ")[0].rstrip("+")
        if number in SUBSTITUTES:
            parts[i] = SUBSTITUTES[number](parts[i])
        elif (verse_id, number, w.get("morph")) == EHYEH:
            parts[i] = parts[i][:-1] + "-" + parts[i][-1]    # Ehyeh: אהי-ה
    # The Tetragrammaton also begins compound names with their own Strong's numbers,
    # like the altar's name in Exodus 17:15, so any part that is exactly the name is replaced too
    return "".join("ה׳" if part == TETRAGRAMMATON else part for part in parts)

verses = []
for book in BOOKS:
    filename = f"{book}.xml"
    if not os.path.exists(filename):              # download each book once
        urllib.request.urlretrieve(URL.format(book), filename)
    root = ET.parse(filename).getroot()
    for verse in root.iter(NS + "verse"):
        # findall only looks at direct children, so words inside notes are skipped
        words = [word_text(w, verse.get("osisID")) for w in verse.findall(NS + "w")]
        verses.append(" ".join(words))

text = "\n".join(verses) + "\n"                   # one verse per line
with open("torah.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Verses:", len(verses))
print("Words:", sum(len(v.split()) for v in verses))
print("Characters in file:", len(text))
print("Distinct characters:", len(set(text)))
print(verses[0])