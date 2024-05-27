# Kodutöö 7 projekt: Pildifailist teksti tuvastamine ja sellest lühikokkuvõtte tegemine 

Koostajad: Kennar Kahju, Mattias Kimst, Joonas Tiitson

## Kirjeldus:
Töö eesmärk oli luua programm mis saab sisendina ette pildifaili, millelt tuvastatakse tekst ning tekstist tehakse etteantud lausete arvuga kokkuvõte. Programm on mõeldud tuvastama ja töötlema eestikeelseid tekste.


## Repositooriumi sisu
- `Kodu7.ipynb`: <b> Jupyter notebook (Colabis käivitatav) pildituvastuse ja kokkuvõtte tegemise lahendustega </b>
- `GUI.py`: Graafilise kasutajaliidese käitatav kood
- `config.py`: Seadistuste fail et muuta tesseract asukohta  
- `näidis_pildid`: Ette antud pildid programmi testimiseks
- `setup`: Esmakordse käivitamise eeljuhised

## Notebooki käivitamine Colab keskkonnas

Ava Kodu7.ipynb Githubis -> ![image](https://github.com/bukyt/MachineLearningProject/assets/68914924/220001d1-e8b0-49af-94bc-a9ea5bcfe42a)

## Lihtne käivitus ilma installimiseta (Windows)

Laadi alla [program.zip](https://github.com/bukyt/MachineLearningProject/releases/), paki see lahti ning käivita `GUI.exe` fail. Veendu, et on lahti pakitud ka lisafailid kaustaga `_internal`.

### 

## Juhised `GUI.py` manuaalselt käivitamiseks

### Esmakordne käivitamine

Kõik vajalikud juhised leiad [setup](/setup/) kaustast.

### Tesseract

Veenduge, et Tesseract-OCR on eelnevalt installitud.

- Kui Tesseract asub mujal kui `config.py` nimetatud asukohas, siis peate muutma vastavalt enda installile.

### Nõuded

- Tesseract nõuab Python 3.11 kasutamist.
- Tesseracti käivitamiseks võivad olla vajalikud administraatori õigused, et vältida pildilt lugemisega seotud õiguste probleeme.

