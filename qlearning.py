import numpy as np
import pygame
import time
import sys
import os
from collections import deque

# ====================================================================
# A. LOGIKA I MAPA (Generator losowy)
# ====================================================================

WYMIARY = (3, 6, 6)  # (Warstwy, Rzędy, Kolumny)
START_POS = (0, 0, 0)
CEL_POS = (2, 5, 5)

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
    kolejka = deque([START_POS])
    odwiedzone = set([START_POS])
    target = CEL_POS

    while kolejka:
        z, r, c = kolejka.popleft()
        if (z, r, c) == target: return True
        for _, (dz, dr, dc) in AKCJE.items():
            nz, nr, nc = z + dz, r + dr, c + dc
            if (0 <= nz < WYMIARY[0] and 0 <= nr < WYMIARY[1] and 0 <= nc < WYMIARY[2]):
                if mapa_testowa[nz, nr, nc] != 1 and (nz, nr, nc) not in odwiedzone:
                    odwiedzone.add((nz, nr, nc))
                    kolejka.append((nz, nr, nc))
    return False


def generuj_losowa_mape():
    próba = 0
    while True:
        próba += 1
        nowa_mapa = np.random.choice([0, 1], size=WYMIARY, p=[0.75, 0.25])
        nowa_mapa[START_POS] = 0
        nowa_mapa[CEL_POS] = 2
        if sprawdz_czy_sciezka_istnieje(nowa_mapa):
            print(f"Wygenerowano poprawną mapę w {próba}. próbie.")
            return nowa_mapa


MAPA = generuj_losowa_mape()
ROZMIAR_MAPY = MAPA.shape
NUM_STANOW = ROZMIAR_MAPY[0] * ROZMIAR_MAPY[1] * ROZMIAR_MAPY[2]


def pos_do_stanu(pos):
    z, r, c = pos
    return z * (ROZMIAR_MAPY[1] * ROZMIAR_MAPY[2]) + r * ROZMIAR_MAPY[2] + c


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
        return 100
    elif val == 1:
        return -10
    else:
        return -1


def nastepny_stan(stan_startowy, akcja):
    z_start, r_start, c_start = stan_do_pos(stan_startowy)

    # --- USUNIĘTO ZASADĘ O OGRANICZENIU WZNOSZENIA ---
    # Agent porusza się standardowo, ogranicza go tylko fizyka (ściany/granice mapy)

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
        while stan_aktualny != stan_celu and kroki < 200:
            kroki += 1
            akcja = wybierz_akcje(stan_aktualny, epsilon)
            stan_nastepny, nagroda = nastepny_stan(stan_aktualny, akcja)
            max_q_nastepny = np.max(Q_TABLE[stan_nastepny, :])
            Q_TABLE[stan_aktualny, akcja] = (1 - ALFA) * Q_TABLE[stan_aktualny, akcja] + \
                                            ALFA * (nagroda + GAMMA * max_q_nastepny)
            stan_aktualny = stan_nastepny
        if epizod % 5000 == 0: print(f"Epizod {epizod}: Epsilon={epsilon:.4f}")
    print("--- TRENING ZAKOŃCZONY ---")


# ====================================================================
# C. WIZUALIZACJA "PANELOWA" (SIDE-BY-SIDE)
# ====================================================================

TILE_SIZE = 60
MARGIN = 20
PANEL_GAP = 50

SZEROKOSC_PANELU = ROZMIAR_MAPY[2] * TILE_SIZE
WYSOKOSC_PANELU = ROZMIAR_MAPY[1] * TILE_SIZE

SZEROKOSC_OKNA = (3 * SZEROKOSC_PANELU) + (2 * PANEL_GAP) + (2 * MARGIN)
WYSOKOSC_OKNA = WYSOKOSC_PANELU + 150

BG_COLOR = (30, 30, 30)
C_GRID = (50, 50, 50)
C_WALL = (80, 80, 80)
C_FREE = (200, 200, 200)
C_START = (50, 50, 200)
C_GOAL = (50, 200, 50)
C_AGENT = (255, 100, 0)
C_ACTIVE_BORDER = (255, 255, 0)
C_INACTIVE_OVERLAY = (0, 0, 0, 150)

os.environ['SDL_AUDIODRIVER'] = 'dsp'
pygame.init()
pygame.font.init()
FONT = pygame.font.SysFont('Arial', 20)
FONT_BIG = pygame.font.SysFont('Arial', 28, bold=True)

EKRAN = pygame.display.set_mode((SZEROKOSC_OKNA, WYSOKOSC_OKNA))
pygame.display.set_caption("3D Drone Navigation - Split View (Slower)")


def rysuj_panel(z_index, offset_x, offset_y, czy_aktywne, agent_pos_local):
    """Rysuje pojedyncze piętro (grid 2D) w zadanej pozycji"""

    tytul = f"PIĘTRO {z_index}"
    kolor_tekstu = (255, 255, 255) if czy_aktywne else (100, 100, 100)
    img_tekst = FONT_BIG.render(tytul, True, kolor_tekstu)
    text_rect = img_tekst.get_rect(center=(offset_x + SZEROKOSC_PANELU // 2, offset_y - 30))
    EKRAN.blit(img_tekst, text_rect)

    rect_panel = pygame.Rect(offset_x - 5, offset_y - 5, SZEROKOSC_PANELU + 10, WYSOKOSC_PANELU + 10)
    kolor_ramki = C_ACTIVE_BORDER if czy_aktywne else (50, 50, 50)
    pygame.draw.rect(EKRAN, kolor_ramki, rect_panel, 3)

    for r in range(ROZMIAR_MAPY[1]):
        for c in range(ROZMIAR_MAPY[2]):
            x = offset_x + c * TILE_SIZE
            y = offset_y + r * TILE_SIZE
            rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)

            typ_pola = MAPA[z_index, r, c]

            if typ_pola == 1:
                kolor = C_WALL
            elif (z_index, r, c) == START_POS:
                kolor = C_START
            elif (z_index, r, c) == CEL_POS:
                kolor = C_GOAL
            else:
                kolor = C_FREE

            pygame.draw.rect(EKRAN, kolor, rect)
            pygame.draw.rect(EKRAN, C_GRID, rect, 1)

            if czy_aktywne and (r, c) == agent_pos_local:
                pygame.draw.circle(EKRAN, C_AGENT, rect.center, TILE_SIZE // 2.5)
                pygame.draw.circle(EKRAN, (255, 255, 255), (rect.centerx - 5, rect.centery - 5), 5)

    if not czy_aktywne:
        s = pygame.Surface((SZEROKOSC_PANELU, WYSOKOSC_PANELU), pygame.SRCALPHA)
        s.fill(C_INACTIVE_OVERLAY)
        EKRAN.blit(s, (offset_x, offset_y))


def rysuj_scene_panelowa(agent_pos):
    EKRAN.fill(BG_COLOR)

    agent_z, agent_r, agent_c = agent_pos
    info = f"Pozycja: (P:{agent_z}, R:{agent_r}, K:{agent_c}) | Cel: {CEL_POS}"
    EKRAN.blit(FONT.render(info, True, (200, 200, 200)), (20, WYSOKOSC_OKNA - 40))

    for z in range(ROZMIAR_MAPY[0]):
        offset_x = MARGIN + (z * (SZEROKOSC_PANELU + PANEL_GAP))
        offset_y = 80
        czy_aktywne = (z == agent_z)
        rysuj_panel(z, offset_x, offset_y, czy_aktywne, (agent_r, agent_c))

    pygame.display.flip()


def testuj_agenta():
    stan_aktualny = pos_do_stanu(START_POS)
    stan_celu = pos_do_stanu(CEL_POS)

    print("\n--- Tryb Testowy (Wolniejszy) ---")

    running = True
    while running and stan_aktualny != stan_celu:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                sys.exit()

        pos_3d = stan_do_pos(stan_aktualny)
        rysuj_scene_panelowa(pos_3d)

        akcja_optymalna = np.argmax(Q_TABLE[stan_aktualny, :])
        stan_nastepny, _ = nastepny_stan(stan_aktualny, akcja_optymalna)

        stan_aktualny = stan_nastepny

        # --- ZMIANA: Spowolnienie do 0.8 sekundy na ruch ---
        time.sleep(0.6)

    if stan_aktualny == stan_celu:
        rysuj_scene_panelowa(stan_do_pos(stan_celu))
        print("SUKCES!")
        time.sleep(3)

    pygame.quit()


# ====================================================================
# D. URUCHOMIENIE
# ====================================================================

if __name__ == "__main__":
    trenuj_agenta()
    testuj_agenta()