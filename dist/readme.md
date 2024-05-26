# Kuidas ise ehitada exe fail
* `pip install pyinstaller`
* Main kausta peal käivita `pyinstaller script.spec`
* Väljund tuleb siia kausta 
* Liiguta exe faili koos `_internal` kaustaga, kuna seal on kõik vajalikud abifailid
* (minul tekkis probleeme _internal/estnltk kaustaga mis oli tühi, peab ise ümber tõstma kui on probleeme)



https://drive.google.com/file/d/1_zIwHdwxLnORVcDIB8UN9PPx3X7wGYnq/view?usp=sharing
