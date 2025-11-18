import numpy as np
import pygame
import time
import sys
import os

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
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0]
    ],
    # Warstwa 1 (Piętro 1)
    [
        [0, 1, 1, 1, 1, 0],
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
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 2]  # Cel
    ]
])

ROZMIAR_MAPY = MAPA.shape  # (3, 6, 6)
NUM_STANOW = ROZMIAR_MAPY[0] * ROZMIAR_MAPY[1] * ROZMIAR_MAPY[2]

AKCJE = {
    0: (0, -1, 0),  # Góra (Północ)
    1: (0, 1, 0),  # Dół (Południe)
    2: (0, 0, -1),  # Lewo (Zachód)
    3: (0, 0, 1),  # Prawo (Wschód)
    4: (1, 0, 0),  # Wznieś się
    5: (-1, 0, 0)  # Opadnij
}
NUM_AKCJI = len(AKCJE)

START_POS = (0, 0, 0)
CEL_POS = (2, 5, 5)


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
    if MAPA[z, r, c] == 2:
        return 100
    elif MAPA[z, r, c] == 1:
        return -10
    else:
        return -1


def nastepny_stan(stan_startowy, akcja):
    z_start, r_start, c_start = stan_do_pos(stan_startowy)
    dz, dr, dc = AKCJE[akcja]
    z_nowy, r_nowy, c_nowy = z_start + dz, r_start + dr, c_start + dc

    if (0 <= z_nowy < ROZMIAR_MAPY[0] and
            0 <= r_nowy < ROZMIAR_MAPY[1] and
            0 <= c_nowy < ROZMIAR_MAPY[2]):

        pos_nowa = (z_nowy, r_nowy, c_nowy)
        if MAPA[z_nowy, r_nowy, c_nowy] == 1:
            return stan_startowy, pobierz_nagrode(pos_nowa)
        return pos_do_stanu(pos_nowa), pobierz_nagrode(pos_nowa)
    else:
        return stan_startowy, -1


# ====================================================================
# B. Q-LEARNING AGENT
# ====================================================================

ALFA = 0.1
GAMMA = 0.9
EPSILON_POCZATKOWY = 1.0
EPSILON_DECAY = 0.9999
LICZBA_EPIZODOW = 15000  # Zmniejszyłem lekko dla szybszego testu

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
        while stan_aktualny != stan_celu and kroki < 500:
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

# Konfiguracja Izometryczna
TILE_WIDTH = 60  # Szerokość kafelka
TILE_HEIGHT = 30  # Wysokość kafelka (spłaszczenie perspektywy)
WALL_HEIGHT = 40  # Wysokość ściany
LAYER_GAP = 120  # Odległość w pionie między piętrami (żeby nie nachodziły na siebie za bardzo)

# Obliczamy rozmiar okna dynamicznie
SZEROKOSC_OKNA = 1000
WYSOKOSC_OKNA = 800

# Kolory
BG_COLOR = (20, 20, 30)
C_FLOOR = (150, 150, 150)
C_WALL_TOP = (200, 100, 100)
C_WALL_SIDE_L = (150, 50, 50)  # Ciemniejszy bok
C_WALL_SIDE_R = (100, 30, 30)  # Najciemniejszy bok
C_START = (50, 50, 200)
C_GOAL = (50, 200, 50)
C_AGENT = (255, 215, 0)  # Złoty dron
C_TEXT = (255, 255, 255)

os.environ['SDL_AUDIODRIVER'] = 'dsp'
pygame.init()
pygame.font.init()
FONT = pygame.font.SysFont('Arial', 16)
FONT_BIG = pygame.font.SysFont('Arial', 24)

EKRAN = pygame.display.set_mode((SZEROKOSC_OKNA, WYSOKOSC_OKNA))
pygame.display.set_caption("3D Drone Navigation - Isometric View")


def konwertuj_iso(x, y, z):
    """
    Konwertuje współrzędne siatki (x, y, z) na współrzędne ekranu 2D (px, py).
    x: kolumna, y: rząd, z: warstwa
    """
    # Centrum ekranu startowe
    start_x = SZEROKOSC_OKNA // 2
    start_y = 150

    # Matematyka izometryczna: obraca siatkę o 45 stopni i spłaszcza
    iso_x = (x - y) * TILE_WIDTH
    iso_y = (x + y) * TILE_HEIGHT

    # Pozycja końcowa uwzględniająca wysokość warstwy (z)
    # Im wyższe z, tym "wyżej" na ekranie (-y)
    screen_x = start_x + iso_x
    screen_y = start_y + iso_y + (z * LAYER_GAP)

    return screen_x, screen_y


def rysuj_szescian(x, y, z, kolor_top, h=WALL_HEIGHT):
    """Rysuje prosty klocek/ścianę w rzucie izometrycznym"""
    px, py = konwertuj_iso(x, y, z)

    # Punkty wierzchołków górnej ściany
    p_top = (px, py - h)
    p_right = (px + TILE_WIDTH, py + TILE_HEIGHT - h)
    p_bottom = (px, py + 2 * TILE_HEIGHT - h)
    p_left = (px - TILE_WIDTH, py + TILE_HEIGHT - h)

    # Punkty podstawy (do rysowania boków)
    p_base_bottom = (px, py + 2 * TILE_HEIGHT)
    p_base_right = (px + TILE_WIDTH, py + TILE_HEIGHT)
    p_base_left = (px - TILE_WIDTH, py + TILE_HEIGHT)

    # 1. Rysowanie boków (cieniowanie)
    # Prawy bok
    pygame.draw.polygon(EKRAN, C_WALL_SIDE_R, [p_right, p_bottom, p_base_bottom, p_base_right])
    # Lewy bok
    pygame.draw.polygon(EKRAN, C_WALL_SIDE_L, [p_left, p_bottom, p_base_bottom, p_base_left])

    # 2. Rysowanie góry
    pygame.draw.polygon(EKRAN, kolor_top, [p_top, p_right, p_bottom, p_left])
    pygame.draw.polygon(EKRAN, (0, 0, 0), [p_top, p_right, p_bottom, p_left], 1)  # Obrys


def rysuj_plytke(x, y, z, kolor):
    """Rysuje płaską podłogę"""
    px, py = konwertuj_iso(x, y, z)
    points = [
        (px, py),
        (px + TILE_WIDTH, py + TILE_HEIGHT),
        (px, py + 2 * TILE_HEIGHT),
        (px - TILE_WIDTH, py + TILE_HEIGHT)
    ]
    pygame.draw.polygon(EKRAN, kolor, points)
    pygame.draw.polygon(EKRAN, (50, 50, 50), points, 1)  # Ciemny obrys


def rysuj_agenta(x, y, z):
    """Rysuje drona jako lewitującą kulę nad daną płytką"""
    px, py = konwertuj_iso(x, y, z)
    # Dron unosi się trochę nad ziemią (-30 pikseli w osi Y ekranu)
    center = (px, int(py + TILE_HEIGHT - 20))

    # Cień drona
    shadow_points = [
        (px, py + TILE_HEIGHT + 5),
        (px + 10, py + TILE_HEIGHT + 10),
        (px, py + TILE_HEIGHT + 15),
        (px - 10, py + TILE_HEIGHT + 10)
    ]
    pygame.draw.polygon(EKRAN, (0, 0, 0), shadow_points)  # Cień

    # Korpus
    pygame.draw.circle(EKRAN, C_AGENT, center, 12)
    pygame.draw.circle(EKRAN, (255, 255, 255), (center[0] - 4, center[1] - 4), 4)  # Błysk


def rysuj_scene_izometryczna(agent_pos):
    EKRAN.fill(BG_COLOR)

    # Rysujemy tekst informacyjny
    tekst = FONT_BIG.render(f"Agent na: z={agent_pos[0]}, y={agent_pos[1]}, x={agent_pos[2]}", True, C_TEXT)
    EKRAN.blit(tekst, (20, 20))
    EKRAN.blit(FONT.render("Piętro 0 (Góra ekranu)", True, C_TEXT), (SZEROKOSC_OKNA - 200, 50))
    EKRAN.blit(FONT.render("Piętro 1", True, C_TEXT), (SZEROKOSC_OKNA - 200, 50 + LAYER_GAP))
    EKRAN.blit(FONT.render("Piętro 2 (Dół ekranu)", True, C_TEXT), (SZEROKOSC_OKNA - 200, 50 + LAYER_GAP * 2))

    # === KLUCZOWE: Pętla renderowania "Od tyłu do przodu" (Painter's Algorithm) ===
    # Musimy rysować warstwy tak, aby obiekty "z przodu" zasłaniały te "z tyłu".

    for z in range(ROZMIAR_MAPY[0]):  # Dla każdego piętra

        # Rysujemy etykietę piętra obok siatki
        label_x, label_y = konwertuj_iso(-1, -1, z)

        for r in range(ROZMIAR_MAPY[1]):  # Dla każdego rzędu
            for c in range(ROZMIAR_MAPY[2]):  # Dla każdej kolumny

                typ_pola = MAPA[z, r, c]

                # Kolor podłogi
                kolor = C_FLOOR
                if (z, r, c) == START_POS:
                    kolor = C_START
                elif (z, r, c) == CEL_POS:
                    kolor = C_GOAL

                # 1. Rysuj Podłogę (wszędzie)
                rysuj_plytke(c, r, z, kolor)

                # 2. Rysuj Ścianę (jeśli jest)
                if typ_pola == 1:
                    rysuj_szescian(c, r, z, C_WALL_TOP)

                # 3. Rysuj Agenta (jeśli tu jest)
                # Sprawdzamy, czy agent jest na tej konkretnej pozycji
                if (z, r, c) == agent_pos:
                    rysuj_agenta(c, r, z)

    pygame.display.flip()


def testuj_agenta():
    stan_aktualny = pos_do_stanu(START_POS)
    stan_celu = pos_do_stanu(CEL_POS)

    print("\n--- Tryb Testowy (Wizualizacja 3D) ---")

    running = True
    while running and stan_aktualny != stan_celu:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                sys.exit()

        pos_3d = stan_do_pos(stan_aktualny)

        # --- NOWA WIZUALIZACJA ---
        rysuj_scene_izometryczna(pos_3d)

        akcja_optymalna = np.argmax(Q_TABLE[stan_aktualny, :])
        stan_nastepny, _ = nastepny_stan(stan_aktualny, akcja_optymalna)

        if stan_nastepny == stan_aktualny:
            print("Agent utknął.")
            break

        stan_aktualny = stan_nastepny
        time.sleep(0.7)  # Trochę wolniej, żeby podziwiać widok

    if stan_aktualny == stan_celu:
        rysuj_scene_izometryczna(stan_do_pos(stan_celu))
        print("SUKCES: Dron dotarł do celu!")
        time.sleep(3)

    pygame.quit()


# ====================================================================
# D. URUCHOMIENIE
# ====================================================================

if __name__ == "__main__":
    trenuj_agenta()
    testuj_agenta()