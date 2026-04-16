# Masterthesis: PETCT_Pipeline

Dieses Repository enthält eine Reihe von Skripten, die zusammen eine Auswertungs-Pipeline für klinische SPECT/CT-Daten bilden (Entwurf mit PET/CT Daten).  

## Grober Überblick: Wie die Skripte zusammenhängen

Ein typischer Ablauf sieht so aus:

1. **CT segmentieren**  
   → `ct_segmentation.py`  
   CT-DICOMs einlesen, nach NIfTI konvertieren, mit TotalSegmentator mehrere Organmasken erzeugen.

2. **(V)OIs definieren & auswerten**
   → `VOI_evaluation.py`
   Berechnung von Kenngrößen (z. B. Voxel, Counts mean / min / max) für 3D-VOIs. Dafür werden die Segmente in das PET (später SPECT) gelegt.

3. **Simulierte Szintigrafie aus PET/CT**  
   → `scintigraphy_simulation.py`  
   Aus 3D-PET/CT-Daten werden planare Projektionen (AP/PA) berechnet, die annähernd einer Gamma-Kamera-Szintigrafie entsprechen. Nutzung des Gammakamera-Modells vom stratos-Ansatz.

4. **ROIs auswerten**  
   → `ROI_valuation.py`   
   Berechnung von Kenngrößen (z. B. Voxel, Counts mean / min / max) für 2D-ROIs. Dafür werden die Segmentmasken in die 2D-Ebene gelegt.

5. **Stratos-Ansatz auf klinische Daten anwenden**  
   → `stratos_applied.py`  
   Aus Organmasken wird eine Systemmatrix aufgebaut; per (nichtnegativer) Lösung des linearen Gleichungssystems werden Organaktivitäten aus den Projektionen geschätzt.




---

## Skripte im Detail

### 1. `ct_segmentation.py`

**Zweck:**  
Automatisierte CT-Vorverarbeitung und Organsegmentierung als Vorbereitung für alle nachfolgenden Schritte.

**Typische Aufgaben:**

- Auffinden und Einlesen einer CT-DICOM-Serie (z. B. Whole-Body CT).
- Konvertieren der DICOM-Serie nach NIfTI.
- Aufruf von **TotalSegmentator** zur automatischen Organ- und Gewebesegmentierung  
  (optional mit „fast“-Modus, z. B. Resampling auf gröbere Auflösung).
- Ablegen der Ergebnisse in einem definierten `results`-Unterordner, z. B.:
  - `ct.nii.gz` – normalisierte CT-Volume  
  - `seg/individual/*.nii.gz` – einzelne Organmasken




### 2. `VOI_evaluation.py`

**Zweck:**
Quantitative Analyse von 3D-Volumes of Interest (VOIs) basierend auf PET- bzw. CT-Daten.

**Typische Aufgaben:**
- Laden der PET- oder CT-NIfTI-Volumes.
- Laden einer oder mehrerer 3D-VOI-Masken.
- Berechnen typischer VOI-Metriken:
	-	Volume (cm³ / ml)
	-	Mean / Max / Median SUV oder Intensitätswerte
	-	Standardabweichung
	-	Histogramme (optional)
-	Export als CSV für statistische Analyse oder KI-Modelle.




### 3. `scintigraphy_simulation.py`

**Zweck:**
Erzeugt synthetische planare Szintigrafieaufnahmen (AP/PA) basierend auf 3D-PET/CT-Daten.

**Kernidee:**
Die PET-Aktivitätsverteilung wird mit einem vereinfachten Gamma-Kamera-Modell entlang einer Projektionsachse auf eine 2D-Ebene projiziert.
Das Skript dient sowohl zur Validierung des Stratos-Ansatzes als auch zur Visualisierung.

**Typische Aufgaben:**
-	Laden der PET- und CT-Volumes (NIfTI oder DICOM).
-	Wahl der Projektionsrichtung (Standard: anterior–posterior und posterior–anterior).
-	Optionale Modellierung von:
   -	linearem Schwächungsmodell
   -	Kollimatorfunktion (simplifiziert)
-	Ausgabe:
   -	2D-Projektionsbilder (NIfTI, PNG oder NumPy-Array)
   -	Optional: Einzelorgan-Projektionen (für Stratos-Validierung)
   -	Optional: Debug-Overlays




### 4. `ROI_evaluation.py`

**Zweck:**
Auswertung von 2D-Region of Interest (ROI), z. B. in einzelnen PET-Slices, CT-Slices oder simulierten planaren Szintigrafieprojektionen.

**Typische Aufgabe:**
-  Laden eines 2D- oder 3D-Bildes (PET, CT, MIP, AP/PA-Projektion).
-	Laden einer oder mehrerer 2D-ROI-Masken.
-	Berechnung zentraler Kennzahlen pro ROI, z. B.:
   -	mittlere und maximale Intensität / Aktivität
   -	Pixelanzahl / Fläche
   -	Standardabweichung, Medianwerte
   -	ggf. einfache Texturmerkmale
-	Export der Resultate (CSV oder JSON).




### 5. `stratos_applied.py`

**Zweck:**
Implementiert die klinische Variante des Stratos-Konzeptes: Schätzung von Organaktivitäten aus planaren AP/PA-Projektionen.

**Kernprinzip:**
-  Jede Organmaske wird einzeln projiziert → ergibt eine Spalte der Systemmatrix A.
-	Die reale (oder simulierte) AP/PA-Projektion ist der Messvektor b.
-	Lösung von:
               A  * x = b              mit Nichtnegativitätsbedingung für die Organaktivitäten.

**Typische Aufgaben:**
-	Laden der PET-/CT-DICOMs.
-	Laden der Organmasken (z. B. aus ct_segmentation.py oder manuell konsolidiert).
-	Aufbau der Systemmatrix A aus projizierten Organvolumina.
-	Lösung mittels:
   -	NNLS (most common)
   -	Tikhonov-Regularisierung (optional)
   -	Lasso- oder ridge-Regression (experimentell)
-	Ausgabe:
   -	geschätzte Organaktivitäten als CSV
   -	rekonstruierte Projektion aus A·x (Vergleich mit gemessener Projektion)
   -	Visualisierung der Residuen