import numpy as np
import pygame
import time
import sys

# ====================================================================
# A. KONFIGURACJA ŚRODOWISKA I MAPY
# ====================================================================

# 0: Wolne pole, 1: Ściana/Przeszkoda, 2: Cel (GOAL)
MAPA = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 2]

])
ROZMIAR_MAPY = MAPA.shape
NUM_STANOW = ROZMIAR_MAPY[0] * ROZMIAR_MAPY[1]

# (0: Góra, 1: Dół, 2: Lewo, 3: Prawo)
AKCJE = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}
NUM_AKCJI = len(AKCJE)

START_POS = (0, 0)
CEL_POS = (5, 5)


# Konwersja (rząd, kolumna) na pojedynczy indeks
def pos_do_stanu(pos):
    return pos[0] * ROZMIAR_MAPY[1] + pos[1]


# Konwersja indeksu (stanu) na pozycję (rząd, kolumna)
def stan_do_pos(stan):
    r = stan // ROZMIAR_MAPY[1]
    c = stan % ROZMIAR_MAPY[1]
    return (r, c)


# Funkcja zwracająca nagrodę dla danej pozycji
def pobierz_nagrode(pos):
    r, c = pos
    if MAPA[r, c] == 2:
        return 100  # Cel
    elif MAPA[r, c] == 1:
        return -10  # Ściana/Pułapka
    else:
        return -1  # Kara za każdy krok


# Funkcja określająca wynik ruchu
def nastepny_stan(stan_startowy, akcja):
    r_start, c_start = stan_do_pos(stan_startowy)
    dr, dc = AKCJE[akcja]
    r_nowy, c_nowy = r_start + dr, c_start + dc

    # Sprawdzenie granic mapy
    if 0 <= r_nowy < ROZMIAR_MAPY[0] and 0 <= c_nowy < ROZMIAR_MAPY[1]:
        pos_nowa = (r_nowy, c_nowy)

        # Jeśli ruch prowadzi do ściany, agent zostaje w miejscu, ale dostaje karę
        if MAPA[r_nowy, c_nowy] == 1:
            return stan_startowy, pobierz_nagrode(pos_nowa)  # Zostaje w tym samym stanie

        return pos_do_stanu(pos_nowa), pobierz_nagrode(pos_nowa)
    else:
        # Agent zostaje w miejscu (próba wyjścia poza mapę)
        return stan_startowy, -1


# ====================================================================
# B. Q-LEARNING AGENT
# ====================================================================

# HIPERPARAMETRY
ALFA = 0.1  # Learning Rate
GAMMA = 0.9  # Discount Factor
EPSILON_POCZATKOWY = 1.0
EPSILON_DECAY = 0.9997
LICZBA_EPIZODOW = 10000

# Inicjalizacja Tablicy Q
Q_TABLE = np.zeros((NUM_STANOW, NUM_AKCJI))


def wybierz_akcje(stan, epsilon):
    if np.random.random() < epsilon:
        # EKPLORACJA
        return np.random.randint(NUM_AKCJI)
    else:
        # EKSPLOATACJA
        return np.argmax(Q_TABLE[stan, :])


def trenuj_agenta():
    global Q_TABLE  # Modyfikujemy globalną tablicę Q
    epsilon = EPSILON_POCZATKOWY
    stan_celu = pos_do_stanu(CEL_POS)

    print(f"--- START TRENINGU ({LICZBA_EPIZODOW} epizodów) ---")

    for epizod in range(LICZBA_EPIZODOW):
        stan_aktualny = pos_do_stanu(START_POS)
        epsilon = max(0.01, epsilon * EPSILON_DECAY)

        kroki = 0
        while stan_aktualny != stan_celu and kroki < 200:  # Limit kroków na epizod
            kroki += 1

            # 1. Wybierz akcję
            akcja = wybierz_akcje(stan_aktualny, epsilon)

            # 2. Wykonaj krok i pobierz r, s'
            stan_nastepny, nagroda = nastepny_stan(stan_aktualny, akcja)

            # 3. AKTUALIZACJA Q-TABLE

            # Max Q(s', a')
            max_q_nastepny = np.max(Q_TABLE[stan_nastepny, :])

            # Nowa wartość Q według wzoru Q-Learning
            q_nowy = (1 - ALFA) * Q_TABLE[stan_aktualny, akcja] + \
                     ALFA * (nagroda + GAMMA * max_q_nastepny)

            Q_TABLE[stan_aktualny, akcja] = q_nowy

            # Przejście do nowego stanu
            stan_aktualny = stan_nastepny

        if epizod % 1000 == 0:
            print(f"Epizod {epizod:5d}: Kroki={kroki:3d}, Epsilon={epsilon:.4f}")

    print("--- TRENING ZAKOŃCZONY ---")


# ====================================================================
# C. WIZUALIZACJA PYGAME I TRYB TESTOWY
# ====================================================================

# KONFIGURACJA WIZUALNA
ROZMIAR_POLA = 80
SZEROKOSC_OKNA = ROZMIAR_MAPY[1] * ROZMIAR_POLA
WYSOKOSC_OKNA = ROZMIAR_MAPY[0] * ROZMIAR_POLA

# Kolory
CZARNY = (0, 0, 0)
BIALY = (255, 255, 255)
ZIELONY = (0, 150, 0)
CZERWONY = (200, 0, 0)
NIEBIESKI = (0, 0, 200)

pygame.init()
EKRAN = pygame.display.set_mode((SZEROKOSC_OKNA, WYSOKOSC_OKNA))
pygame.display.set_caption("Q-Learning Agent w Labiryncie")
ZEGAR = pygame.time.Clock()


def rysuj_labirynt(agent_pos):
    EKRAN.fill(BIALY)

    for r in range(ROZMIAR_MAPY[0]):
        for c in range(ROZMIAR_MAPY[1]):
            rect = pygame.Rect(c * ROZMIAR_POLA, r * ROZMIAR_POLA, ROZMIAR_POLA, ROZMIAR_POLA)

            # Rysowanie siatki
            pygame.draw.rect(EKRAN, CZARNY, rect, 1)

            # Kolorowanie specjalnych pól
            if MAPA[r, c] == 1:  # Ściana
                pygame.draw.rect(EKRAN, CZARNY, rect)
            elif (r, c) == CEL_POS:  # Cel
                pygame.draw.rect(EKRAN, ZIELONY, rect)
            elif (r, c) == START_POS:  # Start
                pygame.draw.rect(EKRAN, NIEBIESKI, rect)

    # Rysowanie Agenta
    agent_center_x = agent_pos[1] * ROZMIAR_POLA + ROZMIAR_POLA // 2
    agent_center_y = agent_pos[0] * ROZMIAR_POLA + ROZMIAR_POLA // 2
    pygame.draw.circle(EKRAN, CZERWONY, (agent_center_x, agent_center_y), ROZMIAR_POLA // 3)

    pygame.display.flip()


def testuj_agenta():
    stan_aktualny = pos_do_stanu(START_POS)
    stan_celu = pos_do_stanu(CEL_POS)

    print("\n--- Tryb Testowy (Wizualizacja optymalnej ścieżki) ---")

    running = True
    while running and stan_aktualny != stan_celu:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        # 1. Konwersja indeksu stanu na pozycję (rząd, kolumna) do rysowania
        r, c = stan_do_pos(stan_aktualny)

        # 2. Wizualizacja bieżącego stanu
        rysuj_labirynt((r, c))

        # 3. Wybór NAJLEPSZEJ akcji (Eksploatacja: wybór max Q)
        akcja_optymalna = np.argmax(Q_TABLE[stan_aktualny, :])

        # 4. Wykonanie kroku
        stan_nastepny, _ = nastepny_stan(stan_aktualny, akcja_optymalna)

        if stan_nastepny == stan_aktualny:
            # Agent nie może się ruszyć (optymalna ścieżka prowadzi w ścianę - błąd?)
            break

        stan_aktualny = stan_nastepny

        # Oczekiwanie, aby móc obserwować ruch
        time.sleep(0.3)

        # Ostatnie rysowanie (na polu celu)
    if stan_aktualny == stan_celu:
        rysuj_labirynt(stan_do_pos(stan_celu))
        print("Agent dotarł do celu!")
        time.sleep(2)

    pygame.quit()


# ====================================================================
# D. URUCHOMIENIE PROJEKTU
# ====================================================================

if __name__ == "__main__":
    # Krok 1: Trening agenta
    trenuj_agenta()

    # Krok 2: Testowanie i wizualizacja optymalnej ścieżki
    testuj_agenta()