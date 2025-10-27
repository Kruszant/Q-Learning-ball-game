import numpy as np
import pygame
import time
import sys
import os  # <--- ZMIANA: Dodane do wyciszenia błędów ALSA w Dockerze

# ====================================================================
# A. KONFIGURACJA ŚRODOWISKA I MAPY (3D)
# ====================================================================

# Użyjemy 3 warstw (pięter) o wymiarach 6x6
# 0: Wolne pole, 1: Ściana/Przeszkoda, 2: Cel (GOAL)
MAPA = np.array([
    # Warstwa 0 (Parter)
    [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],  # Ściana na warstwie 0, pod celem
        [0, 0, 0, 0, 1, 0]
    ],
    # Warstwa 1 (Piętro 1)
    [
        [0, 1, 1, 1, 1, 0],  # Ściany blokujące przejście
        [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0]
    ],
    # Warstwa 2 (Piętro 2)
    [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],  # Wolna droga do celu
        [0, 0, 0, 0, 1, 2]  # Cel jest na warstwie 2
    ]
])  # <--- ZMIANA: Mapa jest teraz 3D

ROZMIAR_MAPY = MAPA.shape  # <--- ZMIANA: (3, 6, 6) - (warstwy, rzędy, kolumny)
NUM_STANOW = ROZMIAR_MAPY[0] * ROZMIAR_MAPY[1] * ROZMIAR_MAPY[2]  # <--- ZMIANA: 3 * 6 * 6 = 108 stanów

# (0: Góra, 1: Dół, 2: Lewo, 3: Prawo, 4: Wznieś się, 5: Opadnij)
AKCJE = {
    0: (0, -1, 0),  # (dz, dr, dc)
    1: (0, 1, 0),
    2: (0, 0, -1),
    3: (0, 0, 1),
    4: (1, 0, 0),  # <--- ZMIANA: Ruch w osi Z (w górę)
    5: (-1, 0, 0)  # <--- ZMIANA: Ruch w osi Z (w dół)
}
NUM_AKCJI = len(AKCJE)  # <--- ZMIANA: 6 akcji

START_POS = (0, 0, 0)  # <--- ZMIANA: (z, r, c)
CEL_POS = (2, 5, 5)  # <--- ZMIANA: (z, r, c)


# Konwersja (z, rząd, kolumna) na pojedynczy indeks
def pos_do_stanu(pos):  # <--- ZMIANA: Logika 3D
    z, r, c = pos
    indeks_warstwy = ROZMIAR_MAPY[1] * ROZMIAR_MAPY[2]
    indeks_rzedu = ROZMIAR_MAPY[2]
    return z * indeks_warstwy + r * indeks_rzedu + c


# Konwersja indeksu (stanu) na pozycję (z, rząd, kolumna)
def stan_do_pos(stan):  # <--- ZMIANA: Logika 3D
    indeks_warstwy = ROZMIAR_MAPY[1] * ROZMIAR_MAPY[2]
    indeks_rzedu = ROZMIAR_MAPY[2]

    z = stan // indeks_warstwy
    r = (stan % indeks_warstwy) // indeks_rzedu
    c = stan % indeks_rzedu
    return (z, r, c)


# Funkcja zwracająca nagrodę dla danej pozycji
def pobierz_nagrode(pos):  # <--- ZMIANA: Logika 3D
    z, r, c = pos
    if MAPA[z, r, c] == 2:
        return 100
    elif MAPA[z, r, c] == 1:
        return -10
    else:
        return -1


# Funkcja określająca wynik ruchu
def nastepny_stan(stan_startowy, akcja):  # <--- ZMIANA: Logika 3D
    z_start, r_start, c_start = stan_do_pos(stan_startowy)
    dz, dr, dc = AKCJE[akcja]
    z_nowy, r_nowy, c_nowy = z_start + dz, r_start + dr, c_start + dc

    # Sprawdzenie granic mapy (3D)
    if (0 <= z_nowy < ROZMIAR_MAPY[0] and
            0 <= r_nowy < ROZMIAR_MAPY[1] and
            0 <= c_nowy < ROZMIAR_MAPY[2]):

        pos_nowa = (z_nowy, r_nowy, c_nowy)

        # Sprawdzenie ścian
        if MAPA[z_nowy, r_nowy, c_nowy] == 1:
            return stan_startowy, pobierz_nagrode(pos_nowa)

        return pos_do_stanu(pos_nowa), pobierz_nagrode(pos_nowa)
    else:
        # Agent zostaje w miejscu (próba wyjścia poza mapę)
        return stan_startowy, -1


# ====================================================================
# B. Q-LEARNING AGENT (Logika pozostaje taka sama)
# ====================================================================

# HIPERPARAMETRY
ALFA = 0.1
GAMMA = 0.9
EPSILON_POCZATKOWY = 1.0
EPSILON_DECAY = 0.9999  # <--- ZMIANA: Wolniejszy spadek, bo więcej stanów
LICZBA_EPIZODOW = 30000  # <--- ZMIANA: Więcej epizodów, bo 108 stanów zamiast 36

# Inicjalizacja Tablicy Q
Q_TABLE = np.zeros((NUM_STANOW, NUM_AKCJI))  # <--- ZMIANA: Rozmiar (108, 6)


def wybierz_akcje(stan, epsilon):
    # Ta funkcja pozostaje bez zmian
    if np.random.random() < epsilon:
        return np.random.randint(NUM_AKCJI)
    else:
        return np.argmax(Q_TABLE[stan, :])


def trenuj_agenta():
    # Ta funkcja pozostaje bez zmian w logice Q-Learning
    global Q_TABLE
    epsilon = EPSILON_POCZATKOWY
    stan_celu = pos_do_stanu(CEL_POS)

    print(f"--- START TRENINGU ({LICZBA_EPIZODOW} epizodów, {NUM_STANOW} stanów) ---")

    for epizod in range(LICZBA_EPIZODOW):
        stan_aktualny = pos_do_stanu(START_POS)
        epsilon = max(0.01, epsilon * EPSILON_DECAY)

        kroki = 0
        while stan_aktualny != stan_celu and kroki < 500:  # <--- ZMIANA: Zwiększony limit kroków
            kroki += 1
            akcja = wybierz_akcje(stan_aktualny, epsilon)
            stan_nastepny, nagroda = nastepny_stan(stan_aktualny, akcja)

            max_q_nastepny = np.max(Q_TABLE[stan_nastepny, :])
            q_nowy = (1 - ALFA) * Q_TABLE[stan_aktualny, akcja] + \
                     ALFA * (nagroda + GAMMA * max_q_nastepny)

            Q_TABLE[stan_aktualny, akcja] = q_nowy
            stan_aktualny = stan_nastepny

        if epizod % 2000 == 0:
            print(f"Epizod {epizod:5d}: Kroki={kroki:3d}, Epsilon={epsilon:.4f}")

    print("--- TRENING ZAKOŃCZONY ---")


# ====================================================================
# C. WIZUALIZACJA PYGAME (Wizualizacja warstwowa)
# ====================================================================

# KONFIGURACJA WIZUALNA
ROZMIAR_POLA = 80
SZEROKOSC_OKNA = ROZMIAR_MAPY[2] * ROZMIAR_POLA  # <--- ZMIANA: Szerokość (kolumny)
WYSOKOSC_OKNA = ROZMIAR_MAPY[1] * ROZMIAR_POLA  # <--- ZMIANA: Wysokość (rzędy)

# Kolory
CZARNY = (0, 0, 0)
BIALY = (255, 255, 255)
ZIELONY = (0, 150, 0)
CZERWONY = (200, 0, 0)
NIEBIESKI = (0, 0, 200)

# Wyciszenie błędów dźwięku ALSA (przydatne w Dockerze)
os.environ['SDL_AUDIODRIVER'] = 'dsp'
pygame.init()
pygame.font.init()  # <--- ZMIANA: Inicjalizacja czcionek
FONT_WARSTWY = pygame.font.SysFont('Arial', 30)

EKRAN = pygame.display.set_mode((SZEROKOSC_OKNA, WYSOKOSC_OKNA))
pygame.display.set_caption("Q-Learning Agent 3D (Wizualizacja Warstwowa)")
ZEGAR = pygame.time.Clock()


def rysuj_labirynt(agent_pos):  # <--- ZMIANA: Przyjmuje (z, r, c)
    EKRAN.fill(BIALY)

    # Pobieramy bieżącą warstwę (piętro), na której jest agent
    aktualna_warstwa = agent_pos[0]

    for r in range(ROZMIAR_MAPY[1]):
        for c in range(ROZMIAR_MAPY[2]):
            rect = pygame.Rect(c * ROZMIAR_POLA, r * ROZMIAR_POLA, ROZMIAR_POLA, ROZMIAR_POLA)

            pygame.draw.rect(EKRAN, CZARNY, rect, 1)

            # Rysujemy elementy TYLKO z bieżącej warstwy
            if MAPA[aktualna_warstwa, r, c] == 1:
                pygame.draw.rect(EKRAN, CZARNY, rect)
            elif (aktualna_warstwa, r, c) == CEL_POS:  # Sprawdzamy pełne współrzędne 3D
                pygame.draw.rect(EKRAN, ZIELONY, rect)
            elif (aktualna_warstwa, r, c) == START_POS:
                pygame.draw.rect(EKRAN, NIEBIESKI, rect)

    # Rysowanie Agenta (używamy tylko r i c, bo warstwę widać po mapie)
    agent_center_x = agent_pos[2] * ROZMIAR_POLA + ROZMIAR_POLA // 2
    agent_center_y = agent_pos[1] * ROZMIAR_POLA + ROZMIAR_POLA // 2
    pygame.draw.circle(EKRAN, CZERWONY, (agent_center_x, agent_center_y), ROZMIAR_POLA // 3)

    # <--- ZMIANA: Wyświetlanie aktualnego piętra
    tekst_warstwy = FONT_WARSTWY.render(f'Piętro: {aktualna_warstwa}', True, CZARNY)
    EKRAN.blit(tekst_warstwy, (10, 10))

    pygame.display.flip()


def testuj_agenta():
    stan_aktualny = pos_do_stanu(START_POS)
    stan_celu = pos_do_stanu(CEL_POS)

    print("\n--- Tryb Testowy (Wizualizacja optymalnej ścieżki 3D) ---")

    running = True
    while running and stan_aktualny != stan_celu:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        # Konwersja stanu na pozycję 3D (z, r, c)
        pos_3d = stan_do_pos(stan_aktualny)

        # Wizualizacja bieżącego stanu (funkcja sama wie, którą warstwę narysować)
        rysuj_labirynt(pos_3d)

        # Wybór NAJLEPSZEJ akcji (Eksploatacja)
        akcja_optymalna = np.argmax(Q_TABLE[stan_aktualny, :])

        # Debugowanie: Pokaż wybraną akcję
        # print(f"Stan: {pos_3d}, Wybrana akcja: {akcja_optymalna}")

        stan_nastepny, _ = nastepny_stan(stan_aktualny, akcja_optymalna)

        if stan_nastepny == stan_aktualny:
            print("Błąd: Agent utknął mimo optymalnej polityki.")
            break

        stan_aktualny = stan_nastepny

        time.sleep(0.3)

    if stan_aktualny == stan_celu:
        rysuj_labirynt(stan_do_pos(stan_celu))
        print("Dron dotarł do celu!")
        time.sleep(2)

    pygame.quit()


# ====================================================================
# D. URUCHOMIENIE PROJEKTU
# ====================================================================

if __name__ == "__main__":
    trenuj_agenta()
    testuj_agenta()