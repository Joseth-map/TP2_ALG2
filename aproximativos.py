import argparse
import inspect
import math
import time
import tracemalloc
import networkx as nx

from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np


# ============================================================
# INSTRUMENTAÇÃO
# ============================================================

@contextmanager
def medir(nome, tempos=None, medir_memoria=False):
    """
    Mede o tempo de execução de um bloco.

    Se medir_memoria=True, registra também o pico observado
    pelo tracemalloc, em bytes.

    Os tempos de chamadas repetidas com o mesmo nome são somados.
    """
    inicio = time.perf_counter()

    iniciou_tracemalloc = False

    if medir_memoria:

        if not tracemalloc.is_tracing():
            tracemalloc.start()
            iniciou_tracemalloc = True

        tracemalloc.reset_peak()

    try:
        yield
    finally:
        duracao = time.perf_counter() - inicio

        if tempos is not None:
            tempos[nome] = tempos.get(nome, 0.0) + duracao

        if medir_memoria:
            if tracemalloc.is_tracing():
                _, pico = tracemalloc.get_traced_memory()

                if tempos is not None:
                    chave = f"pico_memoria_{nome}_MB"

                    pico_mb = pico / (1024 ** 2)

                    tempos[chave] = pico_mb

            if iniciou_tracemalloc:
                tracemalloc.stop()


# ============================================================
# REPRESENTAÇÃO DA INSTÂNCIA
# ============================================================

@dataclass
class Instancia:
    nome: str
    N: int
    edge_weight_type: str
    coords: object = None
    matriz: object = None
    edge_weight_format: str = None


# ============================================================
# LEITURA TSPLIB
# ============================================================

def ler_tsplib(caminho, tempos=None, medir_memoria=False):
    """
    Lê uma instância TSPLIB.

    Para instâncias geométricas, a matriz será construída
    posteriormente por construir_matriz().

    Para EXPLICIT, a matriz é carregada diretamente.
    """

    with medir("ler_tsplib", tempos, medir_memoria):

        nome = None
        N = None
        edge_weight_type = None
        edge_weight_format = None

        coordenadas = []
        pesos = []

        secao = None

        with open(caminho, "r", encoding="utf-8") as arquivo:

            for linha in arquivo:

                linha = linha.strip()

                if not linha:
                    continue

                if linha == "EOF":
                    break

                if linha == "NODE_COORD_SECTION":
                    secao = "NODE_COORD"
                    continue

                if linha == "EDGE_WEIGHT_SECTION":
                    secao = "EDGE_WEIGHT"
                    continue

                if linha.endswith("_SECTION"):
                    secao = "OUTRA"
                    continue

                # Leitura das coordenadas.
                if secao == "NODE_COORD":

                    partes = linha.split()

                    if len(partes) < 3:
                        raise ValueError(
                            f"Coordenada inválida: {linha}"
                        )

                    identificador = int(partes[0])
                    x = float(partes[1])
                    y = float(partes[2])

                    coordenadas.append(
                        (identificador, x, y)
                    )

                # Leitura dos pesos explícitos.
                elif secao == "EDGE_WEIGHT":

                    pesos.extend(
                        int(valor) for valor in linha.split()
                    )

                # Leitura dos cabeçalhos.
                elif secao is None:

                    if ":" in linha:
                        chave, valor = linha.split(":", 1)

                    else:
                        partes = linha.split(maxsplit=1)

                        if len(partes) != 2:
                            continue

                        chave, valor = partes

                    chave = chave.strip().upper()
                    valor = valor.strip()

                    if chave == "NAME":
                        nome = valor

                    elif chave == "DIMENSION":
                        N = int(valor)

                    elif chave == "EDGE_WEIGHT_TYPE":
                        edge_weight_type = valor.upper()

                    elif chave == "EDGE_WEIGHT_FORMAT":
                        edge_weight_format = valor.upper()

        if N is None:
            raise ValueError("DIMENSION não encontrada.")

        if edge_weight_type is None:
            raise ValueError("EDGE_WEIGHT_TYPE não encontrado.")

        coords = None
        matriz = None

        # Instâncias geométricas.
        if edge_weight_type != "EXPLICIT":

            if len(coordenadas) != N:
                raise ValueError(
                    "Quantidade de coordenadas diferente de N."
                )

            # Mantém a ordem de aparição dos vértices.
            coords = np.array(
                [(x, y) for _, x, y in coordenadas],
                dtype=float
            )

        # Instâncias com matriz explícita.
        else:

            if edge_weight_format is None:
                raise ValueError(
                    "EDGE_WEIGHT_FORMAT não encontrado."
                )

            matriz = expandir_matriz_explicita(
                pesos,
                N,
                edge_weight_format
            )

        return Instancia(
            nome=nome or caminho,
            N=N,
            edge_weight_type=edge_weight_type,
            coords=coords,
            matriz=matriz,
            edge_weight_format=edge_weight_format
        )


def expandir_matriz_explicita(pesos, N, formato):
    """
    Converte os pesos da TSPLIB em uma matriz N x N.

    Formatos implementados:
    FULL_MATRIX
    UPPER_ROW
    LOWER_DIAG_ROW
    UPPER_DIAG_ROW
    LOWER_ROW
    """

    D = np.zeros((N, N), dtype=np.int64)

    if formato == "FULL_MATRIX":

        if len(pesos) != N * N:
            raise ValueError("Quantidade incorreta de pesos.")

        D = np.array(
            pesos,
            dtype=np.int64
        ).reshape((N, N))

    elif formato == "UPPER_ROW":

        esperado = N * (N - 1) // 2

        if len(pesos) != esperado:
            raise ValueError("Quantidade incorreta de pesos.")

        k = 0

        for i in range(N):
            for j in range(i + 1, N):

                D[i, j] = pesos[k]
                D[j, i] = pesos[k]

                k += 1

    elif formato == "LOWER_DIAG_ROW":

        esperado = N * (N + 1) // 2

        if len(pesos) != esperado:
            raise ValueError("Quantidade incorreta de pesos.")

        k = 0

        for i in range(N):
            for j in range(i + 1):

                D[i, j] = pesos[k]
                D[j, i] = pesos[k]

                k += 1

    elif formato == "UPPER_DIAG_ROW":

        esperado = N * (N + 1) // 2

        if len(pesos) != esperado:
            raise ValueError("Quantidade incorreta de pesos.")

        k = 0

        for i in range(N):
            for j in range(i, N):

                D[i, j] = pesos[k]
                D[j, i] = pesos[k]

                k += 1

    elif formato == "LOWER_ROW":

        esperado = N * (N - 1) // 2

        if len(pesos) != esperado:
            raise ValueError("Quantidade incorreta de pesos.")

        k = 0

        for i in range(N):
            for j in range(i):

                D[i, j] = pesos[k]
                D[j, i] = pesos[k]

                k += 1

    else:

        raise NotImplementedError(
            f"EDGE_WEIGHT_FORMAT não implementado: {formato}"
        )

    return D


# ============================================================
# CONSTRUÇÃO DA MATRIZ DE DISTÂNCIAS
# ============================================================

def nint(x):
    """
    Arredondamento da TSPLIB:
    inteiro mais próximo, com empate arredondado para cima.
    """
    return math.floor(x + 0.5)


def geo_para_rad(coordenada):
    """
    Converte coordenadas GEO da TSPLIB para radianos.

    A parte decimal representa minutos, não graus decimais.
    """

    deg = int(coordenada)
    minutos = coordenada - deg

    return math.pi * (deg + 5.0 * minutos / 3.0) / 180.0


def construir_matriz(inst, tempos=None, medir_memoria=False):
    """
    Constrói a matriz de distâncias uma única vez.

    A matriz resultante é armazenada em inst.matriz.
    """

    with medir("matriz_distancias", tempos, medir_memoria):

        N = inst.N
        tipo = inst.edge_weight_type

        # EXPLICIT já possui sua matriz.
        if tipo == "EXPLICIT":

            if inst.matriz is None:
                raise ValueError("Matriz EXPLICIT não encontrada.")

            return inst.matriz

        coords = inst.coords

        if coords is None or coords.shape != (N, 2):
            raise ValueError("Coordenadas inválidas.")

        D = np.zeros((N, N), dtype=np.int64)

        # ----------------------------------------------------
        # EUC_2D
        # ----------------------------------------------------

        if tipo == "EUC_2D":

            diferencas = (
                coords[:, np.newaxis, :]
                - coords[np.newaxis, :, :]
            )

            quadrados = np.sum(diferencas ** 2, axis=2)

            distancias = np.sqrt(quadrados)

            D = np.floor(
                distancias + 0.5
            ).astype(np.int64)

        # ----------------------------------------------------
        # CEIL_2D
        # ----------------------------------------------------

        elif tipo == "CEIL_2D":

            diferencas = (
                coords[:, np.newaxis, :]
                - coords[np.newaxis, :, :]
            )

            quadrados = np.sum(diferencas ** 2, axis=2)

            distancias = np.sqrt(quadrados)

            D = np.ceil(distancias).astype(np.int64)

        # ----------------------------------------------------
        # GEO
        # ----------------------------------------------------

        elif tipo == "GEO":

            RRR = 6378.388

            radianos = np.array([
                [
                    geo_para_rad(x),
                    geo_para_rad(y)
                ]
                for x, y in coords
            ])

            for i in range(N):

                for j in range(i + 1, N):

                    lat_i, lon_i = radianos[i]
                    lat_j, lon_j = radianos[j]

                    q1 = math.cos(lon_i - lon_j)
                    q2 = math.cos(lat_i - lat_j)
                    q3 = math.cos(lat_i + lat_j)

                    argumento = (
                        0.5 * (
                            (1 + q1) * q2
                            - (1 - q1) * q3
                        )
                    )

                    # Evita erros numéricos em acos.
                    argumento = max(
                        -1.0,
                        min(1.0, argumento)
                    )

                    distancia = int(
                        RRR * math.acos(argumento) + 1.0
                    )

                    D[i, j] = distancia
                    D[j, i] = distancia

        # ----------------------------------------------------
        # ATT
        # ----------------------------------------------------

        elif tipo == "ATT":

            for i in range(N):

                for j in range(i + 1, N):

                    dx = coords[i, 0] - coords[j, 0]
                    dy = coords[i, 1] - coords[j, 1]

                    r = math.sqrt(
                        (dx ** 2 + dy ** 2) / 10.0
                    )

                    # Regra oficial pseudo-euclidiana TSPLIB.
                    t = nint(r)

                    if t < r:
                        distancia = t + 1
                    else:
                        distancia = t

                    D[i, j] = distancia
                    D[j, i] = distancia

        else:

            raise NotImplementedError(
                f"EDGE_WEIGHT_TYPE não implementado: {tipo}"
            )

        inst.matriz = D

        return D


# ============================================================
# VERIFICAÇÃO DA DESIGUALDADE TRIANGULAR
# ============================================================

def construir_matriz_real(inst):
    """
    Constrói uma matriz de distâncias geométricas
    sem arredondamento.

    Não se aplica a EXPLICIT, pois não existem
    coordenadas geométricas disponíveis.
    """

    N = inst.N
    tipo = inst.edge_weight_type
    coords = inst.coords

    if tipo == "EXPLICIT":
        raise ValueError(
            "Não existem distâncias reais geométricas "
            "para instâncias EXPLICIT."
        )

    D = np.zeros((N, N), dtype=float)

    if tipo in ("EUC_2D", "CEIL_2D"):

        diferencas = (
            coords[:, np.newaxis, :]
            - coords[np.newaxis, :, :]
        )

        D = np.sqrt(
            np.sum(diferencas ** 2, axis=2)
        )

    elif tipo == "ATT":

        for i in range(N):
            for j in range(i + 1, N):

                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]

                distancia = math.sqrt(
                    (dx ** 2 + dy ** 2) / 10.0
                )

                D[i, j] = distancia
                D[j, i] = distancia

    elif tipo == "GEO":

        RRR = 6378.388

        radianos = np.array([
            [geo_para_rad(x), geo_para_rad(y)]
            for x, y in coords
        ])

        for i in range(N):
            for j in range(i + 1, N):

                lat_i, lon_i = radianos[i]
                lat_j, lon_j = radianos[j]

                q1 = math.cos(lon_i - lon_j)
                q2 = math.cos(lat_i - lat_j)
                q3 = math.cos(lat_i + lat_j)

                argumento = 0.5 * (
                    (1 + q1) * q2
                    - (1 - q1) * q3
                )

                argumento = max(
                    -1.0,
                    min(1.0, argumento)
                )

                distancia = RRR * math.acos(argumento)

                D[i, j] = distancia
                D[j, i] = distancia

    else:
        raise NotImplementedError(tipo)

    return D

def verificar_triangular(D):

    D = np.asarray(D)

    lado_esquerdo = D[:, np.newaxis, :]

    lado_direito = (
        D[:, :, np.newaxis]
        + D[np.newaxis, :, :]
    )

    violacoes = lado_esquerdo > lado_direito

    quantidade = int(np.count_nonzero(violacoes))

    return quantidade, quantidade == 0


# ============================================================
# VALIDAÇÃO E CUSTO DE UM TOUR
# ============================================================

def validar_e_custo(
    tour,
    D,
    tempos=None,
    medir_memoria=False
):
    """
    Verifica se tour é uma permutação de 0..N-1.

    Calcula o custo incluindo a aresta de retorno
    do último vértice ao primeiro.
    """

    with medir("validar_e_custo", tempos, medir_memoria):

        D = np.asarray(D)
        N = D.shape[0]

        if D.shape != (N, N):
            raise ValueError("Matriz de distâncias inválida.")

        if len(tour) != N:
            raise ValueError(
                f"Tour deve possuir exatamente {N} vértices."
            )

        if sorted(tour) != list(range(N)):
            raise ValueError(
                "Tour não é uma permutação de 0..N-1."
            )

        custo = 0

        for i in range(N):

            atual = tour[i]
            proximo = tour[(i + 1) % N]

            custo += int(D[atual, proximo])

        return custo


# ============================================================
# GRAU DOS VÉRTICES DA MST
# ============================================================

def calcular_graus(
    arestas,
    N,
    tempos=None,
    medir_memoria=False
):
    """
    Calcula o grau de cada vértice a partir das arestas da MST.
    """

    with medir("grau_mst", tempos, medir_memoria):

        graus = [0] * N

        for u, v in arestas:

            graus[u] += 1
            graus[v] += 1

        return graus


# ============================================================
# MST COM PRIM - O(N²)
# ============================================================

def prim_mst(
    D,
    tempos=None,
    medir_memoria=False
):
    """
    Constrói uma árvore geradora mínima usando Prim.

    Implementação O(N²), sem heap.

    Retorna:
        arestas: lista de pares (u, v)
        graus: grau de cada vértice na MST
    """

    with medir("prim_mst", tempos, medir_memoria):

        D = np.asarray(D)
        N = D.shape[0]

        if D.shape != (N, N):
            raise ValueError("Matriz inválida.")

        if not np.array_equal(D, D.T):
            raise ValueError(
                "Prim implementado para matrizes simétricas."
            )

        # menor[i]: menor custo conhecido para conectar i à MST.
        menor = np.full(N, np.inf)

        # pai[i]: vértice que conecta i à MST.
        pai = np.full(N, -1, dtype=int)

        visitado = np.zeros(N, dtype=bool)

        menor[0] = 0

        arestas = []

        for _ in range(N):

            # Escolhe o vértice não visitado com menor chave.
            candidatos = np.where(
                visitado,
                np.inf,
                menor
            )

            u = int(np.argmin(candidatos))

            if not np.isfinite(menor[u]):
                raise ValueError(
                    "Grafo desconexo: MST não encontrada."
                )

            visitado[u] = True

            if pai[u] != -1:
                arestas.append((int(pai[u]), u))

            # Atualiza os custos dos vértices restantes.
            melhorar = (
                (~visitado)
                & (D[u] < menor)
            )

            menor[melhorar] = D[u, melhorar]

            pai[melhorar] = u

        graus = calcular_graus(
            arestas,
            N,
            tempos,
            medir_memoria
        )

        return arestas, graus


# ============================================================
# ADJACÊNCIA DA ÁRVORE
# ============================================================

def construir_adjacencia(arestas, N):

    adj = [[] for _ in range(N)]

    for u, v in arestas:

        adj[u].append(v)
        adj[v].append(u)

    for vizinhos in adj:
        vizinhos.sort()

    return adj


# ============================================================
# DFS ITERATIVO / PRÉ-ORDEM
# ============================================================

def dfs_pre_ordem(adj, N):
    """
    Percorre a árvore em pré-ordem usando pilha explícita.

    Não utiliza recursão.
    """

    visitado = [False] * N

    pilha = [0]

    tour = []

    while pilha:

        u = pilha.pop()

        if visitado[u]:
            continue

        visitado[u] = True

        tour.append(u)

        # Inverte para visitar primeiro o menor vizinho.
        for v in reversed(adj[u]):

            if not visitado[v]:
                pilha.append(v)

    if len(tour) != N:
        raise ValueError("DFS não alcançou todos os vértices.")

    return tour


# ============================================================
# TWICE-AROUND-THE-TREE
# ============================================================

def twice_around_the_tree(
    D,
    tempos=None,
    medir_memoria=False
):
    """
    Algoritmo Twice-around-the-tree.

    1. Calcula a MST.
    2. Realiza DFS em pré-ordem.
    3. Utiliza atalhos para formar o tour.

    Garantia em métricas:
        custo <= 2 * OPT
    """

    arestas, graus = prim_mst(
        D,
        tempos,
        medir_memoria
    )

    N = len(D)

    adj = construir_adjacencia(arestas, N)

    with medir("atalhos", tempos, medir_memoria):

        # A pré-ordem já visita cada vértice uma única vez.
        tour = dfs_pre_ordem(adj, N)

    custo = validar_e_custo(
        tour,
        D,
        tempos,
        medir_memoria
    )

    return tour, custo


# ============================================================
# EMPARELHAMENTO PERFEITO DE CUSTO MÍNIMO
# ============================================================


def emparelhamento_minimo(
    D,
    vertices_impares,
    tempos=None,
    medir_memoria=False
):
    """
    Encontra um emparelhamento perfeito de custo mínimo
    entre os vértices de grau ímpar.
    """

    with medir("emparelhamento", tempos, medir_memoria):

        G = nx.Graph()

        vertices = list(vertices_impares)

        G.add_nodes_from(vertices)

        # Utiliza diretamente as distâncias originais.
        # O NetworkX minimiza a soma dos pesos.
        for indice, u in enumerate(vertices):

            for v in vertices[indice + 1:]:

                G.add_edge(
                    u,
                    v,
                    weight=int(D[u, v])
                )

        # Compatibilidade entre versões do NetworkX.
        parametros = inspect.signature(
            nx.min_weight_matching
        ).parameters

        if "maxcardinality" in parametros:

            matching = nx.min_weight_matching(
                G,
                maxcardinality=True,
                weight="weight"
            )

        else:

            matching = nx.min_weight_matching(
                G,
                weight="weight"
            )

        if len(matching) * 2 != len(vertices):
            raise RuntimeError(
                "Não foi encontrado emparelhamento perfeito."
            )

        arestas = [
            (int(u), int(v))
            for u, v in matching
        ]

        return arestas



# ============================================================
# CIRCUITO EULERIANO - HIERHOLZER ITERATIVO
# ============================================================

def circuito_euleriano(
    arestas,
    N,
    tempos=None,
    medir_memoria=False
):
    """
    Constrói um circuito euleriano com Hierholzer.

    Arestas paralelas são tratadas individualmente
    por meio de identificadores únicos.
    """

    with medir("euler", tempos, medir_memoria):

        adj = [[] for _ in range(N)]

        # Cada aresta recebe um identificador único.
        for identificador, (u, v) in enumerate(arestas):

            adj[u].append((v, identificador))
            adj[v].append((u, identificador))

        usado = [False] * len(arestas)

        # Índice da próxima aresta a analisar em cada vértice.
        indice = [0] * N

        inicio = next(
            (
                i for i in range(N)
                if adj[i]
            ),
            None
        )

        if inicio is None:
            raise ValueError("Grafo sem arestas.")

        pilha = [inicio]

        circuito = []

        while pilha:

            u = pilha[-1]

            # Ignora arestas já utilizadas.
            while (
                indice[u] < len(adj[u])
                and usado[adj[u][indice[u]][1]]
            ):
                indice[u] += 1

            if indice[u] == len(adj[u]):

                # Não existem mais arestas disponíveis em u.
                circuito.append(pilha.pop())

            else:

                v, identificador = adj[u][indice[u]]

                indice[u] += 1

                if not usado[identificador]:

                    usado[identificador] = True

                    pilha.append(v)

        circuito.reverse()

        if not all(usado):
            raise RuntimeError(
                "Nem todas as arestas foram percorridas."
            )

        return circuito


# ============================================================
# ATALHOS NO CIRCUITO EULERIANO
# ============================================================

def atalhos(
    circuito,
    N,
    tempos=None,
    medir_memoria=False
):
    """
    Percorre o circuito euleriano e mantém apenas
    a primeira ocorrência de cada vértice.
    """

    with medir("atalhos", tempos, medir_memoria):

        visitado = [False] * N

        tour = []

        for v in circuito:

            if not visitado[v]:

                visitado[v] = True

                tour.append(v)

        if len(tour) != N:
            raise RuntimeError(
                "Atalhos não produziram um tour completo."
            )

        return tour


# ============================================================
# CHRISTOFIDES
# ============================================================

def christofides(
    D,
    tempos=None,
    medir_memoria=False
):
    """
    Algoritmo de Christofides.

    1. MST.
    2. Identificação dos vértices ímpares.
    3. Emparelhamento perfeito mínimo.
    4. Construção do multigrafo euleriano.
    5. Circuito de Euler.
    6. Atalhos.

    Garantia em métricas:
        custo <= 1.5 * OPT
    """

    D = np.asarray(D)
    N = len(D)

    # --------------------------------------------------------
    # 1. Árvore geradora mínima
    # --------------------------------------------------------

    arestas_mst, graus = prim_mst(
        D,
        tempos,
        medir_memoria
    )

    # --------------------------------------------------------
    # 2. Vértices de grau ímpar
    # --------------------------------------------------------

    vertices_impares = [
        v
        for v in range(N)
        if graus[v] % 2 == 1
    ]

    # --------------------------------------------------------
    # 3. Emparelhamento perfeito mínimo
    # --------------------------------------------------------

    arestas_matching = emparelhamento_minimo(
        D,
        vertices_impares,
        tempos,
        medir_memoria
    )

    # --------------------------------------------------------
    # 4. Multigrafo euleriano
    # --------------------------------------------------------

    # Não utilizamos set: arestas paralelas precisam ser mantidas.
    arestas_multigrafo = (
        arestas_mst + arestas_matching
    )

    # Verifica se todos os graus são pares.
    graus_euler = [0] * N

    for u, v in arestas_multigrafo:

        graus_euler[u] += 1
        graus_euler[v] += 1

    assert all(g % 2 == 0 for g in graus_euler)

    # --------------------------------------------------------
    # 5. Circuito de Euler
    # --------------------------------------------------------

    circuito = circuito_euleriano(
        arestas_multigrafo,
        N,
        tempos,
        medir_memoria
    )

    # --------------------------------------------------------
    # 6. Atalhos
    # --------------------------------------------------------

    tour = atalhos(
        circuito,
        N,
        tempos,
        medir_memoria
    )

    # --------------------------------------------------------
    # 7. Validação independente
    # --------------------------------------------------------

    custo = validar_e_custo(
        tour,
        D,
        tempos,
        medir_memoria
    )

    return tour, custo


# ============================================================
# DEMONSTRAÇÃO
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Algoritmos aproximativos para TSP."
    )

    parser.add_argument(
        "arquivo",
        nargs="?",
        default="burma14.tsp",
        help="Caminho para a instância TSPLIB."
    )

    parser.add_argument(
        "--memoria",
        action="store_true",
        help="Ativa o rastreamento de memória."
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Leitura e construção da matriz
    # --------------------------------------------------------

    tempos = {}

    inst = ler_tsplib(
        args.arquivo,
        tempos,
        args.memoria
    )

    D = construir_matriz(
        inst,
        tempos,
        args.memoria
    )

    print("=" * 65)
    print("INSTÂNCIA")
    print("=" * 65)

    print(f"Nome: {inst.nome}")
    print(f"Número de vértices: {inst.N}")
    print(f"Tipo: {inst.edge_weight_type}")

    # --------------------------------------------------------
    # Verificação da desigualdade triangular
    # --------------------------------------------------------

    quantidade, metrica = verificar_triangular(
        D,
        tempos,
        args.memoria
    )

    print(f"Violações triangulares: {quantidade}")
    print(f"Matriz métrica: {metrica}")

    # --------------------------------------------------------
    # Ótimo publicado
    # --------------------------------------------------------

    # Ótimo conhecido da instância burma14.
    if inst.nome.lower().startswith("burma14"):
        otimo = 3323

    else:
        raise ValueError(
            "O main desta demonstração possui ótimo configurado "
            "somente para burma14 (3323). "
            "Não atribua um ótimo desconhecido a outra instância."
        )

    print(f"Ótimo publicado: {otimo}")

    # --------------------------------------------------------
    # TWICE-AROUND-THE-TREE
    # --------------------------------------------------------

    tour_tat, custo_tat = twice_around_the_tree(
        D,
        tempos,
        args.memoria
    )

    gap_tat = (
        (custo_tat - otimo) / otimo
    ) * 100

    print("\n" + "=" * 65)
    print("TWICE-AROUND-THE-TREE")
    print("=" * 65)

    print(f"Tour: {tour_tat}")
    print(f"Custo: {custo_tat}")
    print(f"Gap: {gap_tat:.2f}%")

    assert custo_tat >= otimo, (
        "TAT encontrou custo inferior ao ótimo publicado."
    )

    if metrica:

        assert custo_tat <= 2 * otimo, (
            "TAT ultrapassou a garantia de 2x."
        )

    # --------------------------------------------------------
    # CHRISTOFIDES
    # --------------------------------------------------------

    tour_chr, custo_chr = christofides(
        D,
        tempos,
        args.memoria
    )

    gap_chr = (
        (custo_chr - otimo) / otimo
    ) * 100

    print("\n" + "=" * 65)
    print("CHRISTOFIDES")
    print("=" * 65)

    print(f"Tour: {tour_chr}")
    print(f"Custo: {custo_chr}")
    print(f"Gap: {gap_chr:.2f}%")

    assert custo_chr >= otimo, (
        "Christofides encontrou custo inferior ao ótimo publicado."
    )

    if metrica:

        assert custo_chr <= 1.5 * otimo, (
            "Christofides ultrapassou a garantia de 1.5x."
        )

    # --------------------------------------------------------
    # TEMPOS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("TEMPOS DE EXECUÇÃO")
    print("=" * 65)

    for nome, valor in tempos.items():

        if nome.startswith("pico_memoria"):
            print(f"{nome}: {valor:.4f} MB")

        else:
            print(f"{nome}: {valor:.6f} s")

    if not metrica:

        print(
            "\nATENÇÃO: a matriz não é métrica. "
            "As garantias de aproximação não são aplicáveis."
        )


if __name__ == "__main__":
    main()
