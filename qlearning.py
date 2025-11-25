import numpy as np
import pygame
import time
import sys
import os
from collections import deque  # Potrzebne do sprawdzania ścieżki (BFS)

# ====================================================================
# A. KONFIGURACJA ŚRODOWISKA I GENERATOR MAPY (3D)
# ====================================================================

WYMIARY = (3, 6, 6)  # (Warstwy, Rzędy, Kolumny)
START_POS = (0, 0, 0)
CEL_POS = (2, 5, 5)

# (0: Góra, 1: Dół, 2: Lewo, 3: Prawo, 4: Wznieś się, 5: Opadnij)
AKCJE = {
    0: (0, -1, 0),
    1: (0, 1, 0),
    2: (0, 0, -1),
    3: (0, 0, 1),
    4: (1, 0, 0),
    5: (-1, 0, 0)
}
NUM_AKCJI = len(AKCJE)


def sprawdz_czy_sciezka_istnieje(mapa_testowa):
    """
    Używa algorytmu BFS (Breadth-First Search) żeby sprawdzić,
    czy istnieje fizyczne przejście od START do CEL.
    """
    kolejka = deque([START_POS])
    odwiedzone = set([START_POS])

    # Cel w formacie tablicy (z, r, c)
    target = CEL_POS

    while kolejka:
        z, r, c = kolejka.popleft()

        if (z, r, c) == target:
            return True  # Znaleziono drogę!

        # Sprawdź wszystkich sąsiadów (6 kierunków)
        for _, (dz, dr, dc) in AKCJE.items():
            nz, nr, nc = z + dz, r + dr, c + dc

            # Sprawdzenie granic
            if (0 <= nz < WYMIARY[0] and
                    0 <= nr < WYMIARY[1] and
                    0 <= nc < WYMIARY[2]):

                # Jeśli nie jest ścianą (1) i nie był odwiedzony
                if mapa_testowa[nz, nr, nc] != 1 and (nz, nr, nc) not in odwiedzone:
                    odwiedzone.add((nz, nr, nc))
                    kolejka.append((nz, nr, nc))

    return False  # Kolejka pusta, brak drogi


def generuj_losowa_mape():
    """Generuje losową mapę tak długo, aż znajdzie taką z rozwiązaniem."""
    próba = 0
    while True:
        próba += 1
        # 1. Tworzymy pustą macierz 3D
        # Szansa na ścianę to np. 25% (p=[0.75, 0.25])
        nowa_mapa = np.random.choice([0, 1], size=WYMIARY, p=[0.75, 0.25])

        # 2. Czyścimy Start i Cel (muszą być dostępne)
        nowa_mapa[START_POS] = 0
        nowa_mapa[CEL_POS] = 2  # Oznaczamy cel jako 2

        # 3. Sprawdzamy czy da się przejść
        if sprawdz_czy_sciezka_istnieje(nowa_mapa):
            print(f"Wygenerowano poprawną mapę w {próba}. próbie.")
            return nowa_mapa


# === INICJALIZACJA MAPY ===
MAPA = generuj_losowa_mape()
ROZMIAR_MAPY = MAPA.shape
NUM_STANOW = ROZMIAR_MAPY[0] * ROZMIAR_MAPY[1] * ROZMIAR_MAPY[2]


# --- Pozostałe funkcje pomocnicze (bez zmian) ---

def pos_do_stanu(pos):
    z, r, c = pos
    indeks_warstwy = ROZMIAR_MAPY[1] * ROZMIAR_MAPY[2]
    indeks_rzedu = ROZMIAR_MAPY[2]
    return z * indeks_warstwy + r * indeks_rzedu + c


def stan_do_pos(stan):
    indeks_warstwy = ROZMIAR_MAPY[1] * ROZMIAR_MAPY[2]
    indeks_rzedu = ROZMIAR_MAPY[2]
    z = stan // indeks_warstwy
    r = (stan % indeks_warstwy) // indeks_rzedu
    c = stan % indeks_rzedu
    return (z, r, c)


def pobierz_nagrode(pos):
    z, r, c = pos
    val = MAPA[z, r, c]
    if val == 2:
        return 100  # Cel
    elif val == 1:
        return -10  # Ściana
    else:
        return -1  # Puste pole (koszt ruchu)


def nastepny_stan(stan_startowy, akcja):
    z_start, r_start, c_start = stan_do_pos(stan_startowy)
    dz, dr, dc = AKCJE[akcja]
    z_nowy, r_nowy, c_nowy = z_start + dz, r_start + dr, c_start + dc

    if (0 <= z_nowy < ROZMIAR_MAPY[0] and
            0 <= r_nowy < ROZMIAR_MAPY[1] and
            0 <= c_nowy < ROZMIAR_MAPY[2]):

        pos_nowa = (z_nowy, r_nowy, c_nowy)
        if MAPA[z_nowy, r_nowy, c_nowy] == 1:  # Uderzenie w ścianę
            return stan_startowy, pobierz_nagrode(pos_nowa)
        return pos_do_stanu(pos_nowa), pobierz_nagrode(pos_nowa)
    else:
        return stan_startowy, -1  # Wyjście poza mapę


# ====================================================================
# B. Q-LEARNING AGENT
# ====================================================================

ALFA = 0.1
GAMMA = 0.9
EPSILON_POCZATKOWY = 1.0
EPSILON_DECAY = 0.9999
LICZBA_EPIZODOW = 15000

Q_TABLE = np.zeros((NUM_STANOW, NUM_AKCJI))


def wybierz_akcje(stan, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(NUM_AKCJI)
    else:
        return np.argmax(Q_TABLE[stan, :])


def trenuj_agenta():
    global Q_TABLE
    epsilon = EPSILON_POCZATKOWY
    stan_celu = pos_do_stanu(CEL_POS)

    print(f"--- START TRENINGU ({LICZBA_EPIZODOW} epizodów) ---")

    for epizod in range(LICZBA_EPIZODOW):
        stan_aktualny = pos_do_stanu(START_POS)
        epsilon = max(0.01, epsilon * EPSILON_DECAY)
        kroki = 0

        # Pętla jednego epizodu
        while stan_aktualny != stan_celu and kroki < 200:  # Limit kroków per epizod
            kroki += 1
            akcja = wybierz_akcje(stan_aktualny, epsilon)
            stan_nastepny, nagroda = nastepny_stan(stan_aktualny, akcja)

            max_q_nastepny = np.max(Q_TABLE[stan_nastepny, :])
            q_nowy = (1 - ALFA) * Q_TABLE[stan_aktualny, akcja] + \
                     ALFA * (nagroda + GAMMA * max_q_nastepny)

            Q_TABLE[stan_aktualny, akcja] = q_nowy
            stan_aktualny = stan_nastepny

        if epizod % 5000 == 0:
            print(f"Epizod {epizod}: Epsilon={epsilon:.4f}")

    print("--- TRENING ZAKOŃCZONY ---")


# ====================================================================
# C. WIZUALIZACJA IZOMETRYCZNA (PSEUDO-3D)
# ====================================================================

TILE_WIDTH = 60
TILE_HEIGHT = 30
WALL_HEIGHT = 40
LAYER_GAP = 120

SZEROKOSC_OKNA = 1000
WYSOKOSC_OKNA = 800

# Kolory
BG_COLOR = (20, 20, 30)
C_FLOOR = (150, 150, 150)
C_WALL_TOP = (200, 100, 100)
C_WALL_SIDE_L = (150, 50, 50)
C_WALL_SIDE_R = (100, 30, 30)
C_START = (50, 50, 200)
C_GOAL = (50, 200, 50)
C_AGENT = (255, 215, 0)
C_TEXT = (255, 255, 255)

os.environ['SDL_AUDIODRIVER'] = 'dsp'
pygame.init()
pygame.font.init()
FONT = pygame.font.SysFont('Arial', 16)
FONT_BIG = pygame.font.SysFont('Arial', 24)

EKRAN = pygame.display.set_mode((SZEROKOSC_OKNA, WYSOKOSC_OKNA))
pygame.display.set_caption("3D Drone Navigation - Random Maps")


def konwertuj_iso(x, y, z):
    start_x = SZEROKOSC_OKNA // 2
    start_y = 150
    iso_x = (x - y) * TILE_WIDTH
    iso_y = (x + y) * TILE_HEIGHT
    screen_x = start_x + iso_x
    screen_y = start_y + iso_y + (z * LAYER_GAP)
    return screen_x, screen_y


def rysuj_szescian(x, y, z, kolor_top, h=WALL_HEIGHT):
    px, py = konwertuj_iso(x, y, z)
    p_top = (px, py - h)
    p_right = (px + TILE_WIDTH, py + TILE_HEIGHT - h)
    p_bottom = (px, py + 2 * TILE_HEIGHT - h)
    p_left = (px - TILE_WIDTH, py + TILE_HEIGHT - h)
    p_base_bottom = (px, py + 2 * TILE_HEIGHT)
    p_base_right = (px + TILE_WIDTH, py + TILE_HEIGHT)
    p_base_left = (px - TILE_WIDTH, py + TILE_HEIGHT)

    pygame.draw.polygon(EKRAN, C_WALL_SIDE_R, [p_right, p_bottom, p_base_bottom, p_base_right])
    pygame.draw.polygon(EKRAN, C_WALL_SIDE_L, [p_left, p_bottom, p_base_bottom, p_base_left])
    pygame.draw.polygon(EKRAN, kolor_top, [p_top, p_right, p_bottom, p_left])
    pygame.draw.polygon(EKRAN, (0, 0, 0), [p_top, p_right, p_bottom, p_left], 1)


def rysuj_plytke(x, y, z, kolor):
    px, py = konwertuj_iso(x, y, z)
    points = [
        (px, py),
        (px + TILE_WIDTH, py + TILE_HEIGHT),
        (px, py + 2 * TILE_HEIGHT),
        (px - TILE_WIDTH, py + TILE_HEIGHT)
    ]
    pygame.draw.polygon(EKRAN, kolor, points)
    pygame.draw.polygon(EKRAN, (50, 50, 50), points, 1)


def rysuj_agenta(x, y, z):
    px, py = konwertuj_iso(x, y, z)
    center = (px, int(py + TILE_HEIGHT - 20))
    shadow_points = [
        (px, py + TILE_HEIGHT + 5),
        (px + 10, py + TILE_HEIGHT + 10),
        (px, py + TILE_HEIGHT + 15),
        (px - 10, py + TILE_HEIGHT + 10)
    ]
    pygame.draw.polygon(EKRAN, (0, 0, 0), shadow_points)
    pygame.draw.circle(EKRAN, C_AGENT, center, 12)
    pygame.draw.circle(EKRAN, (255, 255, 255), (center[0] - 4, center[1] - 4), 4)


def rysuj_scene_izometryczna(agent_pos):
    EKRAN.fill(BG_COLOR)

    tekst = FONT_BIG.render(f"Agent: {agent_pos} | CEL: {CEL_POS}", True, C_TEXT)
    EKRAN.blit(tekst, (20, 20))
    EKRAN.blit(FONT.render("Piętro 0", True, C_TEXT), (SZEROKOSC_OKNA - 150, 50))
    EKRAN.blit(FONT.render("Piętro 1", True, C_TEXT), (SZEROKOSC_OKNA - 150, 50 + LAYER_GAP))
    EKRAN.blit(FONT.render("Piętro 2", True, C_TEXT), (SZEROKOSC_OKNA - 150, 50 + LAYER_GAP * 2))

    for z in range(ROZMIAR_MAPY[0]):
        for r in range(ROZMIAR_MAPY[1]):
            for c in range(ROZMIAR_MAPY[2]):
                typ_pola = MAPA[z, r, c]
                kolor = C_FLOOR
                if (z, r, c) == START_POS:
                    kolor = C_START
                elif (z, r, c) == CEL_POS:
                    kolor = C_GOAL

                rysuj_plytke(c, r, z, kolor)
                if typ_pola == 1:
                    rysuj_szescian(c, r, z, C_WALL_TOP)
                if (z, r, c) == agent_pos:
                    rysuj_agenta(c, r, z)

    pygame.display.flip()


def testuj_agenta():
    stan_aktualny = pos_do_stanu(START_POS)
    stan_celu = pos_do_stanu(CEL_POS)

    print("\n--- Tryb Testowy ---")

    running = True
    while running and stan_aktualny != stan_celu:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                sys.exit()

        pos_3d = stan_do_pos(stan_aktualny)
        rysuj_scene_izometryczna(pos_3d)

        akcja_optymalna = np.argmax(Q_TABLE[stan_aktualny, :])
        stan_nastepny, _ = nastepny_stan(stan_aktualny, akcja_optymalna)

        if stan_nastepny == stan_aktualny:
            # Mała szansa, że agent wybierze ścianę w początkowych fazach nauki
            # lub jeśli algorytm nie zbiegł idealnie
            pass

        stan_aktualny = stan_nastepny
        time.sleep(0.3)

    if stan_aktualny == stan_celu:
        rysuj_scene_izometryczna(stan_do_pos(stan_celu))
        print("SUKCES: Dron dotarł do celu!")
        time.sleep(3)

    pygame.quit()


if __name__ == "__main__":
    trenuj_agenta()
    testuj_agenta()