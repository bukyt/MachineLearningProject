# Kodutöö 7 projekt: Pildifailist teksti tuvastamine ja sellest lühikokkuvõtte tegemine 

Koostajad: Kennar Kahju, Mattias Kimst, Joonas Tiitson

## Kirjeldus:
Töö eesmärk oli luua programm mis saab sisendina ette pildifaili, millelt tuvastatakse tekst ning tekstist tehakse etteantud lausete arvuga kokkuvõte. Programm on mõeldud tuvastama ja töötlema eestikeelseid tekste.


## Repositooriumi sisu
- `Kodu7.ipynb`: <b> jupyter notebook pildituvastuse ja kokkuvõtte tegemise lahendustega </b>
- `CLIMethods.py`: CLI versioon rakendusest, praegusel hetkel iganenud.  
- `GUI.py`: graafilise kasutajaliidese kood
- `hoidla.txt`: Hoiab koodi, kuid kunagi ei käivitata, loodud debuggimise jooksul.  
- `script.spec`: 
- `setup.bat`: .bat fail mis käivitab mõned käsud windows arvutites rakenduse käivitamiseks
- `setupabi.md`: Abi fail mõne ettepanekuga kuidas rakendust käima saada kui ei kasuta valmis kompileeritud versiooni  
- `variables.config`: config fail et muuta tesseract asukohta  
- `Kuva.png`: näidispilt testimiseks
- `naidis.png`: näidispilt testimiseks
- `placeholder.png`: näidispilt testimiseks

## Juhised GUI.py faili käivitamiseks

### Setupabi.md failis olevate juhiste järgimine

Järgnevad juhised aitavad seadistada ja käivitada projekti `GUI.py` faili oma arvutis.

### Linux Kasutajad

Linuxi kasutajatel tuleb `setup.bat` käske manuaalselt käivitada. 

### Tesseract

Veenduge, et Tesseracti asukoht on `C:\Program Files\Tesseract-OCR\tesseract.exe`.

- Kui Tesseract asub mujal, siis peate muutma faili asukohta real 188 failis `GUI.py`.

### Nõuded

- Tesseracti nõuab Python 3.11 kasutamist.
- Tesseracti käivitamiseks võivad olla vajalikud administraatori õigused, et vältida pildilt lugemisega seotud õiguste probleeme.

