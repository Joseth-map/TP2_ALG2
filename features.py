"""Calcula features geométricas por instância e grava resultados/features.csv.
Rodar na raiz do projeto: python features.py
Instâncias sem coordenadas (EXPLICIT) recebem NA em todas as features."""
import csv
import json
import math
import os

import numpy as np

from aproximativos import ler_tsplib

FEATURES = ["razao_aspecto", "prop_fecho_convexo", "entropia_espacial",
            "densidade_clustering", "grau_max_mst", "cv_dist_vizinho_proximo",
            "dist_media", "dist_desvio"]
CAMPOS = ["instancia", "N", "edge_weight_type", "tem_coordenadas"] + FEATURES


def cruz(o, a, b):
    """Produto vetorial (a-o) x (b-o): >0 se a curva o->a->b vira à esquerda."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def fecho_convexo(pontos):
    """Monotone chain de Andrew. Retorna só os vértices do fecho
    (pontos colineares sobre as arestas não são vértices)."""
    pts = sorted(set(map(tuple, pontos)))
    if len(pts) <= 2:
        return pts
    inferior, superior = [], []
    for p in pts:
        while len(inferior) >= 2 and cruz(inferior[-2], inferior[-1], p) <= 0:
            inferior.pop()
        inferior.append(p)
    for p in reversed(pts):
        while len(superior) >= 2 and cruz(superior[-2], superior[-1], p) <= 0:
            superior.pop()
        superior.append(p)
    return inferior[:-1] + superior[:-1]


def grau_max_prim(D):
    """Grau máximo da MST, com Prim em O(N²) sobre a matriz float D (diagonal zero)."""
    N = len(D)
    usado = np.zeros(N, dtype=bool)
    dist = np.full(N, np.inf)
    pai = np.full(N, -1)
    grau = np.zeros(N, dtype=int)
    dist[0] = 0.0
    for _ in range(N):
        u = int(np.argmin(np.where(usado, np.inf, dist)))
        usado[u] = True
        if pai[u] >= 0:
            grau[u] += 1
            grau[pai[u]] += 1
        atualiza = (~usado) & (D[u] < dist)
        dist[atualiza] = D[u][atualiza]
        pai[atualiza] = u
    return int(grau.max())


def features(P):
    """P: array N×2 de coordenadas. Retorna dict com as features."""
    N = len(P)
    xmin, ymin = P.min(axis=0)
    xmax, ymax = P.max(axis=0)
    w, h = xmax - xmin, ymax - ymin
    area = w * h

    # Razão de aspecto do bounding box (NA se a bounding box for degenerada)
    aspecto = max(w, h) / min(w, h) if min(w, h) > 0 else None

    # Proporção de pontos que são vértices do fecho convexo
    prop_fecho = len(fecho_convexo(P)) / N

    # Entropia normalizada da distribuição em grade 4×4 (0 = concentrado, 1 = uniforme)
    H, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=4)
    p = H[H > 0] / N
    entropia = float(-(p * np.log(p)).sum() / np.log(16))

    # Distâncias euclidianas em float (features, não a matriz D do algoritmo)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=-1))
    iu = np.triu_indices(N, 1)
    dist_media = float(D[iu].mean())
    dist_desvio = float(D[iu].std())

    # Distância ao vizinho mais próximo: heterogeneidade local e Clark-Evans
    Dz = D.copy()
    np.fill_diagonal(Dz, np.inf)
    nn = Dz.min(axis=1)
    cv = float(nn.std() / nn.mean()) if nn.mean() > 0 else None
    # Clark-Evans R: <1 agrupado, ~1 aleatório, >1 disperso
    clark_evans = float(nn.mean() / (0.5 * math.sqrt(area / N))) if area > 0 else None

    # Grau máximo da MST (hubs)
    Dprim = D.copy()
    np.fill_diagonal(Dprim, 0.0)
    grau = grau_max_prim(Dprim)

    return {
        "razao_aspecto": aspecto,
        "prop_fecho_convexo": prop_fecho,
        "entropia_espacial": entropia,
        "densidade_clustering": clark_evans,
        "grau_max_mst": grau,
        "cv_dist_vizinho_proximo": cv,
        "dist_media": dist_media,
        "dist_desvio": dist_desvio,
    }


def main():
    with open("experimentos.json", encoding="utf-8") as f:
        cfg = json.load(f)
    os.makedirs("resultados", exist_ok=True)
    caminho = "resultados/features.csv"

    with open(caminho, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS)
        escritor.writeheader()
        for e in cfg["instancias"]:
            linha = {"instancia": e["nome"], "N": e["N"],
                     "edge_weight_type": e["edge_weight_type"],
                     "tem_coordenadas": e["tem_coordenadas"]}
            if e["tem_coordenadas"]:
                inst = ler_tsplib(f"instancias/_extraidos/{e['nome']}.tsp")
                linha.update(features(np.asarray(inst.coords, dtype=float)))
            else:
                linha.update({k: None for k in FEATURES})
            linha = {k: ("NA" if v is None else v) for k, v in linha.items()}
            escritor.writerow(linha)
            print(linha["instancia"], "ok")
    print(f"Gravado em {caminho}")


if __name__ == "__main__":
    main()