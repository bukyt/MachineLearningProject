# Keskkonna ettevalmistamine programmi lokaalseks käivitamiseks

Siin on juhised esmakordselt lokaalseks käivitamiseks.  
Kui käivitate Colab notebooki, ei ole vaja seda järgida.

## Mida vaja?
* Python 3.11
* Tesseract-OCR eraldi installida 
    * Windows [installi siit](https://github.com/UB-Mannheim/tesseract/wiki)
    * Linux `sudo apt install tesseract-ocr`
* Järgnevad Pythoni paketid:
    * pytesseract
    * estnltk
    * sentence-transformers
        * Lihtsaks installimiseks on lisatud `setup.bat` (Windows), mis installib kõik ära
* Muuda `config.py` failis ära `tesseract_cmd` asukoht enda Tesseract-OCR installi asukohaks
    * Suure tõenäosusega Windowsi peal ei ole see vajalik


