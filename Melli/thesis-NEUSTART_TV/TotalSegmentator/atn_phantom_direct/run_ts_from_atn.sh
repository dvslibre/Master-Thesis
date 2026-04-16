#!/usr/bin/env bash
# set -x   # (optional) Debug-Ausgabe: zeigt jeden Befehl vor Ausführung
set -e     # brich bei *erstem* Fehler ab (sicherer als still weiterzulaufen)

# ============================================================
#               1) BASIS-PFADE UND EINGABE
# ============================================================

# Projekt-/Datenordner, in dem die .bin-Datei liegt
ATN_DIR="/home/mnguest12/projects/thesis/TotalSegmentator/atn_phantom_direct"

# Rohdaten (XCAT-µ als Binärdatei)
IN_BIN="${ATN_DIR}/phantom_02_ct.par_atn_1.bin"

# Tag/Name für alle Ausgaben (landet in Dateinamen)
TAG="xcat02"

# Zielordner für Outputs (NIfTI, PNG, JSON, ggf. Segmentierung)
OUT_DIR="${ATN_DIR}/runs_${TAG}"

# Ordner sicher anlegen (existiert er schon, passiert nichts)
mkdir -p "${OUT_DIR}"

# (optional) ins Datenverzeichnis wechseln – relative Pfade werden einfacher
cd "${ATN_DIR}"

# ============================================================
#               2) GEOMETRIE DER BINÄRDATEI
# ============================================================
# So „würfeln“ wir die flache Binärdatei in ein 3D-Volumen:
SHAPE_XYZ="256,256,650"  # Voxelanzahl in (X,Y,Z)
BIN_ORDER="XYZ"          # Achsreihenfolge *in der Datei* (Permutation von X,Y,Z)
ARRAY_ORDER="F"          # Speicherlayout: "F"=Fortran/MATLAB, "C"=C/NumPy-Default

# ============================================================
#               3) PHYSIK / EINHEITEN
# ============================================================
# Voxelabstände in mm (Z,Y,X) – „Lineal“ für NIfTI
SPACING="1.0,1.0,1.0"

# Referenz-µ für Wasser [1/cm]; wird für HU-Umrechnung gebraucht:
# HU = 1000 * (µ - µ_Wasser) / µ_Wasser
MU_WATER="0.19286726"    # Beispielwert (70 keV)

# Einheit der Rohdaten in der .bin:
# - "per_mm": das Python-Script rechnet intern auf 1/cm um (x10)
# - "per_cm": nichts zu tun
IN_UNITS="per_mm"

# ============================================================
#               4) RESAMPLING & TOTALSEGMENTATOR
# ============================================================
# Isotropes Resampling (z.B. auf 1.0 mm Kantenlänge) – gut für TS/Anzeige
RESAMPLE_MM="1.0"

# TotalSegmentator einschalten?
RUN_TS=1                 # 1 = ja, 0 = nein

# Schnellmodus (etwas ungenauer, aber flotter)
TS_FAST=0                # 1 = schnell, 0 = volle Qualität

# Hinweis: Dein Python-Script kennt *kein* --ts-tta Argument.
# Wenn du TTA möchtest, gib es ggf. über TS_ARGS an (z.B. "--tta"), abhängig von deiner TS-Version.
TS_TTA=0                 # behalten wir als Schalter, aber unten *nicht* an CMD anhängen

# Zusätzliche TotalSegmentator-Argumente (frei erweiterbar, z.B. "--task body")
TS_ARGS=""

# ============================================================
#               5) PREVIEW / ANZEIGE (OVERLAY)
# ============================================================
# Welche Schnittebene im PNG?
PLANE="coronal"          # "coronal", "sagittal" oder "axial"

# Bilddrehung (Anzahl 90°-Drehungen gegen den Uhrzeigersinn, 1=90°, 2=180°, ...)
ROTATE="1"

# Spiegelungen (nur für die *Vorschau*, nicht die Daten)
FLIP_LR=0                # 1 = links/rechts spiegeln
FLIP_UD=1                # 1 = oben/unten spiegeln (z.B. Kopf oben)

# Legende einblenden?
LEGEND=0
LEGEND_MAX=20            # max. Anzahl Einträge in der Legende
LEGEND_OUTSIDE=0         # 1 = Legende als eigene Spalte rechts

# Colorbar (HU-Skala) einblenden?
HU_PNG=1

# Fensterung/Clipping der HU-Werte (für Anzeige *und* Begrenzung im Script)
CLIP_MIN="-1024"
CLIP_MAX="2000"

# Deckkraft der Segmentierungsfüllung (0..1); höher = deckender
FILL_ALPHA="0.35"

# ============================================================
#               6) HU-STATISTIKEN
# ============================================================
STATS=1  # 1 = JSON mit globalen/per-Label Stats schreiben

# ============================================================
#               7) BEFEHL ZUSAMMENBAUEN
# ============================================================
# Wir nutzen ein Bash-Array CMD=() – das schützt automatisch korrekt vor Leerzeichen in Pfaden.
CMD=(python3 atn_to_hu_nifti.py
  --in-bin "${IN_BIN}"
  --shape-xyz "${SHAPE_XYZ}"
  --bin-order "${BIN_ORDER}"
  --array-order "${ARRAY_ORDER}"
  --in-units "${IN_UNITS}"
  --spacing "${SPACING}"
  --mu-water "${MU_WATER}"
  --out-dir "${OUT_DIR}"
  --tag "${TAG}"
  --preview                      # PNG-Vorschau erzeugen
  --plane "${PLANE}"
  --rotate "${ROTATE}"
  --clip-hu-min "${CLIP_MIN}"
  --clip-hu-max "${CLIP_MAX}"
  --fill-alpha "${FILL_ALPHA}"
)

# Optionale Schalter nur anhängen, wenn aktiv
[[ "${FLIP_LR}" == "1" ]]        && CMD+=(--flip-lr)
[[ "${FLIP_UD}" == "1" ]]        && CMD+=(--flip-ud)

# PNG-Extras
[[ "${LEGEND}" == "1" ]]         && CMD+=(--legend --legend-max "${LEGEND_MAX}")
[[ "${LEGEND_OUTSIDE}" == "1" ]] && CMD+=(--legend-outside)
[[ "${HU_PNG}" == "1" ]]         && CMD+=(--hu-png)

# Resampling
# (wenn RESAMPLE_MM leer wäre, würde -n false sein, aber hier ist "1.0" gesetzt)
[[ -n "${RESAMPLE_MM}" ]]        && CMD+=(--resample-mm "${RESAMPLE_MM}")

# TotalSegmentator
[[ "${RUN_TS}" == "1" ]]         && CMD+=(--run-ts)
[[ "${TS_FAST}" == "1" ]]        && CMD+=(--ts-fast)

# WICHTIG: Dein Python-Script kennt KEIN --ts-tta → nicht anhängen, sonst argparse-Fehler
# [[ "${TS_TTA}" == "1" ]]       && CMD+=(--ts-tta)   # ❌ würde fehlschlagen

# Zusätzliche TS-Argumente (falls angegeben), z.B. "--task body" oder "--tta" je nach TS-Version
[[ -n "${TS_ARGS}" ]]            && CMD+=(--ts-args "${TS_ARGS}")

# (Optional) eigene Label-Map angeben (JSON/CSV mit id->name)
# CMD+=(--ts-labels "/pfad/zu/labels.json")

# HU-Statistiken
[[ "${STATS}" == "1" ]]          && CMD+=(--stats)

# ============================================================
#               8) AUSGABE UND AUSFÜHRUNG
# ============================================================
# Zum Nachvollziehen einmal den kompletten Befehl ausgeben
echo "[RUN]" "${CMD[@]}"

# Jetzt wirklich starten
"${CMD[@]}"

echo "[OK] Fertig. Outputs liegen in: ${OUT_DIR}"