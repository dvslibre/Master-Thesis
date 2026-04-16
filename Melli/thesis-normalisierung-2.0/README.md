# Thesis Repositories – Übersicht

Diese Übersicht beschreibt kurz die Rolle der wichtigsten Repos. Details stehen jeweils in den projektinternen READMEs.

## Repos

- [`PhantomGenerator/`](./PhantomGenerator/): Erzeugt synthetische Phantom-Daten; aktuell in der Praxis primär für qplanar verwendet.
- [`Data_Processing/`](./Data_Processing/): Datenaufbereitung und Vorverarbeitung (z. B. Konvertierung, Normalisierung, Vorbereitung für Training/Evaluation).
- [`pieNeRF/`](./pieNeRF/): NeRF-basierte Rekonstruktion bzw. Modelltraining auf den vorbereiteten Daten.
- [`STRATOS/`](./STRATOS/): Matlab Code for the analytical approach (QPlanar based).

## Empfohlene Reihenfolge

1. (qplanar) Daten mit `PhantomGenerator` erzeugen.
2. Mit `Data_Processing` aufbereiten.
3. Mit `pieNeRF` trainieren/auswerten.
