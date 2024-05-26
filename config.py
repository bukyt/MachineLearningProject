import pytesseract
import os

# Tesseract-OCR käitatava faili asukoht
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Eestikeelse tähemärgi tuvastuse mudeli kaust
os.environ['TESSDATA_PREFIX'] = r'tessdata'