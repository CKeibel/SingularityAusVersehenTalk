# JavaLand 2022 - Singularity aus Versehen [Code]
## Setup
<ul>
<li>Python 3.9</li>
</ul>
Zum installieren der notwendigen Pakete im Verzeichnis folgenden Befehl eingeben:<br>

```pip install -r requirements.txt``` <br><br>
<a href="http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html">Hier</a> sind die Roms der Atari Spiele zu finden. 
Die entpackten Roms werden mit folgendem Befehl hinzugefügt:

```python -m atari_py.import_roms <PATH_TO_ROMS_DIR>```

## Struktur
In 
```main.py``` 
befindet sich der Code um einen Agenten zu laden und spielen zu lassen. 
In 
```wrappers.py```
befinden sich wrapper, die das Training eines Agenten vereinfachen.
(Mehr dazu <a href="https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf">hier</a>)<br>
<br>
```Tutorial.ipynb``` ist eine Anleitung zum trainieren eines Agenten mittles Deep Q-Network.
Fall keine starke Nvidia GPU vorhanden, lassen sich Agenten gut mittels Jupyter Notebooks via
<a href="https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjTsbr8t7T2AhXIRfEDHUVDDdsQFnoECAcQAQ&url=https%3A%2F%2Fresearch.google.com%2Fcolaboratory%2F&usg=AOvVaw38J01zt_Dlb6pQ1fe6FGrI">Google Colab</a> trainieren. </br>
</br>
Weitere "Tutorial"-Notebooks zum Thema Deep Reinforcement Learning können auch
<a href="https://github.com/CKeibel/Deep_Reinforcement_Learning">hier</a> gefunden werden.