# Projekt SIWR - Śledzenie osób z użyciem modeli probabilistycznych

## Opis projektu
Projekt na przedmiot Sztuczna Inteligencja w Robotyce polegał na wykorzystaniu modeli probabilistycznych do śledzenia bounding boxów, w których znajdują się osoby. Celem projektu było zastosowanie metod probabilistycznych oraz grafowych do analizy i śledzenia osób na zdjęciach.

## Etapy projektu
1. Pomniejszenie bounding boxów: Pierwszym krokiem było pomniejszenie bounding boxów w celu ograniczenia obszaru tła i skupienia się na obszarach zawierających osoby.
2. Obliczenie histogramów: Na pomniejszonych bounding boxach obliczono histogramy, które stanowiły reprezentację kolorów w danym obszarze. 
3. Wykorzystanie metod grafowych: Do modelowania i analizy danych użyto metod grafowych. Stworzono graf czynników (FactorGraph), który uwzględniał zależności między bounding boxami, informacjami z histogramów i właściwościach boxów.


## Technologie i narzędzia
- Język programowania: Python, Pgmpy
- Biblioteki: OpenCV, NumPy


## Uruchamianie projektu
W celu włączenia skryptu należy w terminalu wpisać:

    python3 main.py file_path
 Gdzie filepath to ścieżka prowadzaca do pliku txt oraz podrzędnego folderu *frames*



