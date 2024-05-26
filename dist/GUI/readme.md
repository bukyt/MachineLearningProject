# Kuidas ise ehitada exe fail
* `pip install pyinstaller`
* Main kausta peal käivita `pyinstaller script.spec`
* Väljund tuleb siia kausta 
* Liiguta exe faili koos `_internal` kaustaga, kuna seal on kõik vajalikud abifailid
* (minul tekkis probleeme _internal/estnltk kaustaga mis oli tühi, peab ise ümber tõstma kui on probleeme)


# Esialgne katse :
https://drive.google.com/file/d/17rNN2cDiDGogl1pytQjJ3RXgstPWKvtc/view?usp=drive_link
