import gc
import itertools
import random
import time
import tracemalloc
import gzip
import numpy as np

from pathlib import Path
from types import SimpleNamespace

from aproximativos import (
    ler_tsplib,
    construir_matriz,
    verificar_triangular,
    validar_e_custo,
    prim_mst,
    twice_around_the_tree,
    christofides,
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

INF = 10**12
TIMEOUT_PADRAO = 1800

ESTRATEGIAS = [
    "trivial",
    "vizinho",
    "fecho",
    "TAT",
    "CHR",
]


# ============================================================
# INSTRUMENTAÇÃO
# ============================================================

def _registrar_tempo(tempos, nome, inicio):
    decorrido = time.perf_counter() - inicio

    registro = next(
        (r for r in tempos if r["subrotina"] == nome),
        None
    )

    if registro is None:
        registro = {
            "subrotina": nome,
            "tempo_total_s": 0.0,
            "chamadas": 0,
        }

        tempos.append(registro)

    registro["tempo_total_s"] += decorrido
    registro["chamadas"] += 1


def _iniciar_medicao(nome=None):
    return time.perf_counter()

def _finalizar_medicao(tempos, nome, inicio):
    _registrar_tempo(tempos, nome, inicio)
    

def preparar_instancias():
    pasta_base = Path(__file__).resolve().parent
    pasta_extraidos = pasta_base / "instancias" / "_extraidos"

    pasta_extraidos.mkdir(exist_ok=True)

    for arquivo_gz in pasta_base.glob("*.tsp.gz"):
        nome_tsp = arquivo_gz.name[:-3]
        destino = pasta_extraidos / nome_tsp

        if not destino.exists():
            with gzip.open(arquivo_gz, "rt", encoding="utf-8") as origem:
                conteudo = origem.read()

            destino.write_text(conteudo, encoding="utf-8")

    return pasta_extraidos

# ============================================================
# REDUÇÃO DE MATRIZ
# ============================================================

def matriz_reducao(M, tempos):
    inicio = _iniciar_medicao("matriz_reducao")

    try:
        reducao_total = 0

        # Redução das linhas
        min_linhas = M.min(axis=1)

        mascara_linhas = (
            (min_linhas > 0) &
            (min_linhas < INF)
        )

        np.subtract(
            M,
            min_linhas[:, None],
            out=M,
            where=mascara_linhas[:, None] & (M < INF)
        )

        reducao_total += int(
            min_linhas[mascara_linhas].sum()
        )

        # Redução das colunas sobre a matriz já reduzida
        min_colunas = M.min(axis=0)

        mascara_colunas = (
            (min_colunas > 0) &
            (min_colunas < INF)
        )

        np.subtract(
            M,
            min_colunas[None, :],
            out=M,
            where=mascara_colunas[None, :] & (M < INF)
        )

        reducao_total += int(
            min_colunas[mascara_colunas].sum()
        )

        return M, int(reducao_total)

    finally:
        _finalizar_medicao(
            tempos,
            "matriz_reducao",
            inicio
        )


# ============================================================
# MATRIZ RESIDUAL E LOWER BOUND
# ============================================================

def normalizar_tour(tour, N):
    tour = list(tour)

    # Remove a repetição da cidade inicial, caso o tour esteja fechado.
    if len(tour) == N + 1 and tour[0] == tour[-1]:
        tour.pop()

    if len(tour) != N or len(set(tour)) != N:
        raise ValueError(
            f"Tour inválido: esperado {N} vértices distintos, "
            f"recebidos {len(tour)}."
        )

    return tour

def construir_matriz_residual(caminho, D):
    M = np.asarray(D, dtype=np.int64).copy()

    N = M.shape[0]
    ultima = caminho[-1]

    np.fill_diagonal(M, INF)

    M[caminho[:-1], :] = INF
    M[:, caminho[1:]] = INF

    if len(caminho) < N:
        M[ultima, 0] = INF

    return M


def calculo_lb(caminho, custo_acumulado, D, tempos):
    """
    Calcula o limite inferior de um caminho parcial.

    LB = custo acumulado + redução da matriz residual.
    """

    inicio = _iniciar_medicao()

    try:

        M = construir_matriz_residual(caminho, D)

        M, reducao = matriz_reducao(M, tempos)

        lb = custo_acumulado + reducao

        return M, lb

    finally:
        _finalizar_medicao(
            tempos,
            "calculo_lb",
            inicio
        )


# ============================================================
# WARM-START: FECHO CONVEXO
# ============================================================

def produto_vetorial(O, A, B):
    """
    Calcula o produto vetorial 2D utilizado no fecho convexo.
    """

    return (
        (A[0] - O[0]) * (B[1] - O[1])
        - (A[1] - O[1]) * (B[0] - O[0])
    )


def fecho_convexo(coords):
    """
    Calcula o fecho convexo pela monotone chain de Andrew.

    Retorna os índices dos vértices do fecho, sem repetir
    o primeiro vértice no final.
    """

    pontos = [
        (float(coords[i][0]), float(coords[i][1]), i)
        for i in range(len(coords))
    ]

    pontos.sort()

    if len(pontos) <= 1:
        return [p[2] for p in pontos]

    inferior = []

    for p in pontos:

        while (
            len(inferior) >= 2
            and produto_vetorial(
                inferior[-2],
                inferior[-1],
                p
            ) <= 0
        ):
            inferior.pop()

        inferior.append(p)

    superior = []

    for p in reversed(pontos):

        while (
            len(superior) >= 2
            and produto_vetorial(
                superior[-2],
                superior[-1],
                p
            ) <= 0
        ):
            superior.pop()

        superior.append(p)

    # Remove os extremos duplicados.
    hull = inferior[:-1] + superior[:-1]

    return [p[2] for p in hull]


def insercao_feixo(coords, D):
    """
    Constrói um tour por inserção incremental no fecho convexo.

    A cada passo escolhe o par (vértice, aresta) de menor
    incremento de custo.
    """

    N = len(D)

    rota = fecho_convexo(coords)

    restantes = set(range(N)) - set(rota)

    while restantes:

        melhor_incremento = float("inf")
        melhor_vertice = None
        melhor_posicao = None

        for v in sorted(restantes):

            for i in range(len(rota)):

                a = rota[i]
                b = rota[(i + 1) % len(rota)]

                incremento = (
                    int(D[a][v])
                    + int(D[v][b])
                    - int(D[a][b])
                )

                candidato = (
                    incremento,
                    v,
                    i
                )

                melhor = (
                    melhor_incremento,
                    melhor_vertice
                    if melhor_vertice is not None
                    else INF,
                    melhor_posicao
                    if melhor_posicao is not None
                    else INF
                )

                if candidato < melhor:

                    melhor_incremento = incremento
                    melhor_vertice = v
                    melhor_posicao = i

        # Insere o vértice entre a e b.
        rota.insert(melhor_posicao + 1, melhor_vertice)

        restantes.remove(melhor_vertice)

    return rota + [rota[0]]


# ============================================================
# WARM-START: DEMAIS ESTRATÉGIAS
# ============================================================

def vizinho_mais_proximo(D):
    """
    Constrói um tour pelo vizinho mais próximo.

    O algoritmo sempre começa no vértice 0.
    """

    N = len(D)

    visitados = {0}
    caminho = [0]

    atual = 0

    while len(visitados) < N:

        candidatos = [
            j for j in range(N)
            if j not in visitados
        ]

        proximo = min(
            candidatos,
            key=lambda j: (D[atual][j], j)
        )

        caminho.append(proximo)

        visitados.add(proximo)

        atual = proximo

    caminho.append(0)

    return caminho


def warmstart(inst, D, estrategia, tempos):
    """
    Executa a estratégia escolhida e calcula um UB inicial.

    O custo de qualquer tour é recalculado por validar_e_custo.
    """

    inicio = _iniciar_medicao()

    try:

        motivo_NA = None

        N = len(D)

        # ----------------------------------------------------
        # Estratégia trivial
        # ----------------------------------------------------

        if estrategia == "trivial":

            tour = list(range(N)) + [0]

        # ----------------------------------------------------
        # Vizinho mais próximo
        # ----------------------------------------------------

        elif estrategia == "vizinho":

            tour = vizinho_mais_proximo(D)

        # ----------------------------------------------------
        # Inserção pelo fecho convexo
        # ----------------------------------------------------

        elif estrategia == "fecho":

            if inst.edge_weight_type == "EXPLICIT":

                tour = None

                motivo_NA = "nao_aplicavel"

            else:

                tour = insercao_feixo(
                    inst.coords,
                    D
                )

        # ----------------------------------------------------
        # Twice Around the Tree
        # ----------------------------------------------------

        elif estrategia == "TAT":

            tour, _ = twice_around_the_tree(D)

        # ----------------------------------------------------
        # Christofides
        # ----------------------------------------------------

        elif estrategia == "CHR":

            tour, _ = christofides(D)

        else:

            raise ValueError(
                f"Estrategia desconhecida: {estrategia}"
            )

        if tour is None:

            ub = float("inf")

        else:

            tour = list(tour)

            # Garante a representação fechada do ciclo.
            if tour[0] != tour[-1]:
                tour.append(tour[0])

            # Nunca utilizamos o custo retornado pela heurística.
            tour = normalizar_tour(tour, N)
            ub = validar_e_custo(tour, D)

        return tour, ub, motivo_NA

    finally:

        _finalizar_medicao(
            tempos,
            "warmstart",
            inicio
        )


# ============================================================
# EXPANSÃO DE NÓ
# ============================================================

def expansao_no(no, D, UB, tempos):
    """
    Gera os filhos de um nó.

    Para cada cidade ainda não visitada:
        1. Adiciona a aresta ao caminho.
        2. Atualiza o custo real acumulado.
        3. Calcula a matriz residual e seu LB.
        4. Descarta o filho se LB >= UB.

    Retorna:
        filhos: lista de filhos não podados;
        podados: quantidade de filhos descartados.
    """

    inicio = _iniciar_medicao()

    try:

        caminho = no["caminho"]
        custo = no["custo"]

        ultima = caminho[-1]

        N = len(D)

        visitados = set(caminho)

        filhos = []

        podados = 0

        for j in range(N):

            if j in visitados:
                continue

            # Custo real da nova aresta.
            novo_custo = custo + int(D[ultima][j])

            novo_caminho = caminho + [j]

            # O último vértice ainda poderá retornar a 0.
            M, lb = calculo_lb(
                novo_caminho,
                novo_custo,
                D,
                tempos
            )

            filho = {
                "caminho": novo_caminho,
                "custo": novo_custo,
                "lb": lb,
                "profundidade": len(novo_caminho) - 1,
            }

            if lb >= UB:

                podados += 1

                continue

            filhos.append(filho)

        # Ordem crescente de LB.
        filhos.sort(
            key=lambda filho: (
                filho["lb"],
                filho["caminho"][-1]
            )
        )

        return filhos, podados

    finally:

        _finalizar_medicao(
            tempos,
            "expansao_no",
            inicio
        )


# ============================================================
# VERIFICAÇÃO TRIANGULAR
# ============================================================

def verificar_triangular_real(inst):
    """
    Conta violações triangulares usando distâncias euclidianas
    sem arredondamento.

    Aplicável somente a EUC_2D e CEIL_2D, conforme a
    especificação deste experimento.

    Para GEO, ATT e EXPLICIT, retorna None.
    """

    tipo = str(inst.edge_weight_type).strip().upper()

    if tipo not in ("EUC_2D", "CEIL_2D"):
        return None

    coords = inst.coords

    N = len(coords)

    D_float = np.zeros((N, N), dtype=float)

    for i in range(N):

        for j in range(N):

            dx = float(coords[i][0] - coords[j][0])
            dy = float(coords[i][1] - coords[j][1])

            D_float[i][j] = np.sqrt(
                dx**2 + dy**2
            )

    violacoes = 0

    tolerancia = 1e-12

    for i in range(N):

        for j in range(N):

            if i == j:
                continue

            for k in range(N):

                if k == i or k == j:
                    continue

                if (
                    D_float[i][j]
                    > D_float[i][k] + D_float[k][j] + tolerancia
                ):
                    violacoes += 1

    return violacoes


def obter_violacoes(inst, D):
    """
    Retorna as violações triangulares reais e arredondadas.
    """

    violacoes_arredondadas, _ = verificar_triangular(D)

    violacoes_reais = verificar_triangular_real(inst)

    return (
        int(violacoes_reais) if violacoes_reais is not None else None,
        int(violacoes_arredondadas)
    )


# ============================================================
# BUSCA BRANCH-AND-BOUND
# ============================================================

def busca_total(
    D,
    caminho_inicial,
    ub_inicial,
    limite_s,
    inicio_total,
    tempos
):
    """
    Executa o Branch-and-Bound com DFS e pilha explícita.

    Contadores:

    nos_expandidos:
        Nós cujos filhos foram gerados.

    nos_podados:
        Filhos descartados porque LB >= UB.

    profundidade_max:
        Maior profundidade de qualquer nó alcançado.
        A raiz possui profundidade zero.

    lower_bound_final:
        Menor LB dos nós que permanecem na pilha ao terminar.
        Se a pilha estiver vazia, recebe o UB final.

    melhor_ub_final:
        Menor custo de tour completo encontrado.
    """

    inicio = _iniciar_medicao()

    try:

        N = len(D)

        # ----------------------------------------------------
        # Inicialização da raiz
        # ----------------------------------------------------

        M_raiz, lb_raiz = calculo_lb(
            [0],
            0,
            D,
            tempos
        )

        raiz = {
            "caminho": [0],
            "custo": 0,
            "lb": lb_raiz,
            "profundidade": 0,
        }

        pilha = [raiz]

        melhor_ub = ub_inicial

        melhor_tour = caminho_inicial

        nos_expandidos = 0
        nos_podados = 0
        profundidade_max = 0

        status = "OK"
        motivo_NA = None

        # ----------------------------------------------------
        # DFS
        # ----------------------------------------------------

        while pilha:

            if time.perf_counter() - inicio >= limite_s:
                status = "NA"
                motivo_NA = "timeout"
                break

            no = pilha.pop()

            caminho = no["caminho"]

            profundidade = no["profundidade"]

            # Atualiza a profundidade alcançada.
            profundidade_max = max(
                profundidade_max,
                profundidade
            )

            # ------------------------------------------------
            # Folha: todas as cidades foram visitadas.
            # ------------------------------------------------

            if len(caminho) == N:

                custo_tour = (
                    no["custo"]
                    + int(D[caminho[-1]][0])
                )

                if custo_tour < melhor_ub:

                    melhor_ub = custo_tour

                    melhor_tour = caminho + [0]

                continue

            # ------------------------------------------------
            # Timeout: verificado no início da expansão.
            # ------------------------------------------------

            if time.perf_counter() - inicio_total > limite_s:

                # Recoloca o nó na pilha para preservar
                # seu LB no cálculo do lower_bound_final.
                pilha.append(no)

                status = "NA"
                motivo_NA = "timeout"

                break

            # ------------------------------------------------
            # Expansão
            # ------------------------------------------------

            nos_expandidos += 1

            profundidade_max = max(
                profundidade_max,
                profundidade + 1
            )

            filhos, podados = expansao_no(
                no,
                D,
                melhor_ub,
                tempos
            )

            nos_podados += podados

            # ------------------------------------------------
            # Empilhamento
            # ------------------------------------------------

            # Os filhos já estão em ordem crescente de LB.
            # Inverter a ordem garante que o menor LB seja
            # retirado primeiro da pilha.
            for filho in reversed(filhos):

                profundidade_max = max(
                    profundidade_max,
                    filho["profundidade"]
                )

                pilha.append(filho)

        # ----------------------------------------------------
        # Lower bound final
        # ----------------------------------------------------

        if pilha:

            lower_bound_final = min(
                no["lb"] for no in pilha
            )

        else:

            lower_bound_final = melhor_ub

        tour_resultado = None

        if status == "OK":
            # Remove eventual fechamento para normalizar o ciclo.
            tour_resultado = normalizar_tour(melhor_tour, N)

            # Rotaciona o ciclo para começar na cidade 0.
            posicao_zero = tour_resultado.index(0)

            tour_resultado = (
                tour_resultado[posicao_zero:]
                + tour_resultado[:posicao_zero]
            )

            # Fecha o ciclo.
            tour_resultado.append(0)

        resultado = {
            "status": status,
            "motivo_NA": motivo_NA,
            "tour": tour_resultado,
            "custo": (
                melhor_ub
                if status == "OK"
                else None
            ),
            "nos_expandidos": nos_expandidos,
            "nos_podados": nos_podados,
            "profundidade_max": profundidade_max,
            "lower_bound_final": lower_bound_final,
            "melhor_ub_final": (
                melhor_ub
                if melhor_ub != float("inf")
                else None
            ),
        }

        return resultado

    finally:

        _finalizar_medicao(
            tempos,
            "busca_total",
            inicio
        )


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def branch_and_bound(
    inst,
    D,
    estrategia,
    limite_s=1800,
    tempos=None,
    modo_medicao="tempo"
):
    """
    Executa o Branch-and-Bound exato para TSP.

    Parâmetros:
        inst: instância retornada por ler_tsplib.
        D: matriz inteira de custos.
        estrategia: trivial, vizinho, fecho, TAT ou CHR.
        limite_s: limite de execução em segundos.
        tempos: lista opcional para instrumentação.
        modo_medicao: modo de medição ("tempo" ou "memoria").

    Retorna os campos exigidos pelo execucoes.csv.
    """

    if modo_medicao not in ("tempo", "memoria"):
        raise ValueError("Modo de medição inválido.")

    medir_memoria = modo_medicao == "memoria"

    if medir_memoria:
        if tracemalloc.is_tracing():
            raise RuntimeError(
                "tracemalloc já está ativo. "
                "Execute a medição de memória isoladamente."
            )

        tracemalloc.start()

    if estrategia not in ESTRATEGIAS:
        raise ValueError(
            f"Estrategia invalida: {estrategia}"
        )

    if tempos is None:
        tempos = []

    inicio_total = _iniciar_medicao()

    inicio_cronometro = time.perf_counter()

    resultado = None

    try:

        # ----------------------------------------------------
        # Verificações iniciais
        # ----------------------------------------------------

        D = np.asarray(D, dtype=np.int64)

        N = len(D)

        if D.shape != (N, N):
            raise ValueError("A matriz D deve ser quadrada.")

        if N < 2:
            raise ValueError("O TSP exige pelo menos 2 cidades.")

        if np.any(D < 0):
            raise ValueError("Os custos devem ser não negativos.")

        if np.any(D >= INF):
            raise ValueError(
                "Os custos devem ser menores que o sentinela INF."
            )

        # ----------------------------------------------------
        # Warm-start
        # ----------------------------------------------------

        inicio_heuristica = time.perf_counter()

        tour_inicial, ub_inicial, motivo_warmstart = warmstart(
            inst,
            D,
            estrategia,
            tempos
        )

        tempo_heuristica = (
            time.perf_counter() - inicio_heuristica
        )

        # ----------------------------------------------------
        # Busca
        # ----------------------------------------------------

        resultado_busca = busca_total(
            D,
            tour_inicial,
            ub_inicial,
            limite_s,
            inicio_cronometro,
            tempos
        )

        status = resultado_busca["status"]

        motivo_NA = resultado_busca["motivo_NA"]

        # A não aplicabilidade da heurística não impede
        # a conclusão exata da busca.
        if (
            status == "OK"
            and motivo_warmstart is not None
        ):
            motivo_NA = motivo_warmstart

        # ----------------------------------------------------
        # Verificação triangular
        # ----------------------------------------------------

        violacoes_reais, violacoes_arredondadas = (
            obter_violacoes(inst, D)
        )

        # ----------------------------------------------------
        # Resultado
        # ----------------------------------------------------

        resultado = {
            "custo": resultado_busca["custo"],
            "tour": resultado_busca["tour"],
            "status": status,
            "motivo_NA": motivo_NA,
            "tempo_total_s": (
                time.perf_counter() - inicio_cronometro
            ),
            "tempo_heuristica_s": tempo_heuristica,
            "nos_expandidos": resultado_busca["nos_expandidos"],
            "nos_podados": resultado_busca["nos_podados"],
            "profundidade_max": resultado_busca["profundidade_max"],
            "lower_bound_final": resultado_busca["lower_bound_final"],
            "melhor_ub_final": resultado_busca["melhor_ub_final"],
            "pico_memoria_MB": None,
            "violacoes_reais": violacoes_reais,
            "violacoes_arredondadas": violacoes_arredondadas,
            "tempos": tempos,
        }

        return resultado

    finally:

        # ----------------------------------------------------
        # Liberação explícita de memória
        # ----------------------------------------------------

        # A pilha e as matrizes dos nós são liberadas ao
        # terminar busca_total. Aqui também forçamos a coleta
        # dos objetos temporários restantes.

        if medir_memoria:
            _, pico = tracemalloc.get_traced_memory()

            if resultado is not None:
                resultado["pico_memoria_MB"] = (
                    pico / (1024 ** 2)
                )

            tracemalloc.stop()

        gc.collect()

        _finalizar_medicao(
            tempos,
            "busca_total_externa",
            inicio_total
        )

        # Observação:
        # gc.collect() não substitui o isolamento em processo
        # separado para controlar memória e interromper
        # execuções que ultrapassem os limites do sistema.


# ============================================================
# VALIDAÇÃO POR FORÇA BRUTA
# ============================================================

def forca_bruta(D):
    """
    Calcula o ótimo por enumeração de todas as permutações.

    Utilizado exclusivamente para instâncias pequenas.
    """

    N = len(D)

    melhor_custo = float("inf")
    melhor_tour = None

    for permutacao in itertools.permutations(
        range(1, N)
    ):

        tour = [0] + list(permutacao) + [0]

        tour = normalizar_tour(tour, N)
        custo = validar_e_custo(tour, D)

        if custo < melhor_custo:

            melhor_custo = custo
            melhor_tour = tour

    return melhor_custo, melhor_tour


def gerar_instancia_aleatoria(N, rng):
    """
    Gera uma instância EUC_2D com coordenadas inteiras pequenas.

    A matriz D é construída pela mesma função utilizada
    nas instâncias TSPLIB.
    """

    coords = np.array(
        [
            [
                rng.randint(0, 30),
                rng.randint(0, 30)
            ]
            for _ in range(N)
        ],
        dtype=float
    )

    inst = SimpleNamespace(
        nome=f"aleatoria_{N}",
        N=N,
        edge_weight_type="EUC_2D",
        coords=coords,
        matriz=None,
        edge_weight_format=None,
    )

    D = construir_matriz(inst)

    return inst, np.asarray(D, dtype=np.int64)


def validar_forca_bruta(
    quantidade=50,
    seed=42
):
    """
    Testa o B&B contra força bruta em 50 instâncias.

    Cada instância é executada com as cinco estratégias.

    Verifica:
        1. Concordância com força bruta.
        2. Concordância entre as estratégias.
        3. Coerência dos contadores.
    """

    rng = random.Random(seed)

    resultados_validacao = []

    for indice in range(quantidade):

        N = rng.randint(4, 9)

        inst, D = gerar_instancia_aleatoria(
            N,
            rng
        )

        # Ótimo exato por força bruta.
        custo_otimo, tour_otimo = forca_bruta(D)

        custos_estrategias = {}

        for estrategia in ESTRATEGIAS:

            tempos = []

            resultado = branch_and_bound(
                inst,
                D,
                estrategia,
                limite_s=1800,
                tempos=tempos
            )

            assert resultado["status"] == "OK", (
                f"Timeout na instância {indice}, "
                f"estratégia {estrategia}"
            )

            # 1. Concordância com força bruta.
            assert resultado["custo"] == custo_otimo, (
                f"Erro na instância {indice}, "
                f"estratégia {estrategia}: "
                f"B&B={resultado['custo']}, "
                f"força bruta={custo_otimo}"
            )

            custos_estrategias[estrategia] = (
                resultado["custo"]
            )

            # 2. Contadores.
            assert resultado["nos_expandidos"] >= 0

            assert resultado["profundidade_max"] <= N

            assert (
                resultado["lower_bound_final"]
                <= resultado["custo"]
            )

        # 3. Concordância entre as cinco estratégias.
        assert len(set(custos_estrategias.values())) == 1

        resultados_validacao.append({
            "instancia": inst.nome,
            "N": N,
            "custo_otimo": custo_otimo,
            "estrategias": custos_estrategias,
        })

        print(
            f"[OK] Instância {indice + 1}/{quantidade} "
            f"| N={N} "
            f"| Ótimo={custo_otimo}"
        )

    print("\nValidação por força bruta concluída.")

    return resultados_validacao


# ============================================================
# VALIDAÇÃO DE TIMEOUT E CONTADORES
# ============================================================

def validar_timeout_e_contadores():
    """
    Verifica a coerência dos contadores em uma execução OK
    e em uma execução interrompida por timeout.
    """

    rng = random.Random(123)

    inst, D = gerar_instancia_aleatoria(
        8,
        rng
    )

    # --------------------------------------------------------
    # Execução normal
    # --------------------------------------------------------

    resultado_ok = branch_and_bound(
        inst,
        D,
        "trivial",
        limite_s=1800
    )

    assert resultado_ok["status"] == "OK"

    assert resultado_ok["nos_expandidos"] >= 0

    assert resultado_ok["profundidade_max"] <= inst.N

    assert (
        resultado_ok["lower_bound_final"]
        <= resultado_ok["custo"]
    )

    # --------------------------------------------------------
    # Execução interrompida
    # --------------------------------------------------------

    resultado_na = branch_and_bound(
        inst,
        D,
        "trivial",
        limite_s=0
    )

    assert resultado_na["status"] == "NA"

    assert resultado_na["motivo_NA"] == "timeout"

    assert resultado_na["custo"] is None

    assert resultado_na["tour"] is None

    assert resultado_na["nos_expandidos"] >= 0

    assert resultado_na["profundidade_max"] <= inst.N

    print("\nValidação de timeout e contadores concluída.")


# ============================================================
# INSTÂNCIAS TSPLIB PUBLICADAS
# ============================================================

def validar_otimos_publicados(
    diretorio="."
):
    """
    Executa o B&B com estratégia trivial nas instâncias:

        burma14  -> 3323
        gr17     -> 2085
        ulysses16 -> 6859

    Os valores publicados são referências externas.
    A comparação é feita sempre sobre a matriz D.
    """

    referencias = {
        "burma14": 3323,
        "gr17": 2085,
        "ulysses16": 6859,
    }

    for nome, otimo_publicado in referencias.items():

        caminho = f"{diretorio}/{nome}.tsp"

        try:

            inst = ler_tsplib(caminho)

            D = construir_matriz(inst)

            resultado = branch_and_bound(
                inst,
                D,
                estrategia="trivial",
                limite_s=1800
            )

            print(f"\nInstância: {nome}")

            print(
                f"Ótimo publicado: {otimo_publicado}"
            )

            print(
                f"Custo encontrado: {resultado['custo']}"
            )

            print(
                f"Status: {resultado['status']}"
            )

            if resultado["status"] == "OK":

                print(
                    f"Coincide: "
                    f"{resultado['custo'] == otimo_publicado}"
                )

            else:

                print(
                    "Comparação inconclusiva: "
                    "a busca não terminou."
                )

        except FileNotFoundError:

            print(
                f"\n[AVISO] Arquivo não encontrado: {caminho}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VALIDAÇÃO DO BRANCH-AND-BOUND")
    print("=" * 60)

    pasta_instancias = preparar_instancias()

    # 1. Força bruta + concordância entre estratégias.
    validar_forca_bruta(
        quantidade=50,
        seed=42
    )

    # 2. Coerência dos contadores e timeout.
    validar_timeout_e_contadores()

    # 3. Comparação com ótimos publicados.
    pasta_instancias = preparar_instancias()

    validar_otimos_publicados(
        diretorio=str(pasta_instancias)
    )

    print("\nTodas as validações foram executadas.")


if __name__ == "__main__":
    main()