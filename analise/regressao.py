from pathlib import Path
import warnings
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, permutation_test
from scipy.optimize import curve_fit

# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ = Path(__file__).resolve().parents[1]

ARQUIVO_EXECUCOES = RAIZ / "resultados" / "execucoes.csv"
ARQUIVO_FEATURES = RAIZ / "resultados" / "features.csv"
ARQUIVO_PROFILING = RAIZ / "resultados" / "profiling.csv"
ARQUIVO_AMBIENTE = RAIZ / "resultados" / "ambiente.csv"
ARQUIVO_RELATORIO = RAIZ / "resultados" / "analise_final.txt"

PASTA_SAIDA = RAIZ / "resultados" / "resultados_regressao"
PASTA_GRAFICOS = PASTA_SAIDA / "graficos"

PASTA_SAIDA.mkdir(exist_ok=True)
PASTA_GRAFICOS.mkdir(exist_ok=True)

ESTRATEGIAS_BNB = [
    "trivial",
    "vizinho",
    "fecho",
    "TAT",
    "CHR",
]

ALGORITMOS_APROXIMADOS = ["TAT", "CHR"]

# Quantidade mínima de pontos distintos para tentar ajustar
# uma curva. Mesmo com esse mínimo, o resultado é exploratório.
MIN_PONTOS_REGRESSAO = 4

# Análise multivariada só será tentada quando houver amostras
# suficientes em relação ao número de variáveis.
MIN_AMOSTRAS_POR_FEATURE = 5

SEED = 42

plt.rcParams.update({
    "figure.figsize": (9, 6),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})


# ============================================================
# UTILITÁRIOS
# ============================================================

def salvar_csv(df, nome):
    caminho = PASTA_SAIDA / nome
    df.to_csv(caminho, index=False, encoding="utf-8-sig")
    print(f"[OK] {caminho.name}")


def salvar_grafico(nome):
    caminho = PASTA_GRAFICOS / nome
    plt.tight_layout()
    plt.savefig(caminho, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] Gráfico: {caminho.name}")


def converter_numericas(df, colunas):
    for coluna in colunas:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(
                df[coluna],
                errors="coerce"
            )
    return df


def exigir_colunas(df, colunas, arquivo):
    ausentes = [c for c in colunas if c not in df.columns]

    if ausentes:
        raise ValueError(
            f"O arquivo {arquivo} não possui as colunas: {ausentes}"
        )


def mediana_segura(serie):
    serie = pd.to_numeric(serie, errors="coerce").dropna()

    if serie.empty:
        return np.nan

    return serie.median()


def r2_manual(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)

    if ss_tot == 0:
        return np.nan

    return 1 - ss_res / ss_tot


def rmse_manual(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.sqrt(np.mean((y_real - y_pred) ** 2))


def normalizar_texto(serie):
    return (
        serie.astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# LEITURA E LIMPEZA
# ============================================================

def carregar_execucoes():
    if not ARQUIVO_EXECUCOES.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {ARQUIVO_EXECUCOES}"
        )

    # Importante:
    # keep_default_na=False impede que o pandas transforme
    # automaticamente a string literal "NA" em NaN.
    df = pd.read_csv(
        ARQUIVO_EXECUCOES,
        keep_default_na=False,
        na_values=["", "NA"],
    )

    colunas_obrigatorias = [
        "instancia",
        "N",
        "edge_weight_type",
        "algoritmo",
        "estrategia_warmstart",
        "modo",
        "repeticao",
        "tempo_total_s",
        "tempo_heuristica_s",
        "custo",
        "gap",
        "nos_expandidos",
        "nos_podados",
        "pico_memoria_MB",
        "violacoes_reais",
        "violacoes_arredondadas",
        "status",
        "motivo_NA",
    ]

    exigir_colunas(
        df,
        colunas_obrigatorias,
        ARQUIVO_EXECUCOES.name,
    )

    colunas_numericas = [
        "N",
        "repeticao",
        "custo",
        "gap",
        "tempo_total_s",
        "tempo_heuristica_s",
        "nos_expandidos",
        "nos_podados",
        "profundidade_max",
        "lower_bound_final",
        "melhor_ub_final",
        "pico_memoria_MB",
        "violacoes_reais",
        "violacoes_arredondadas",
    ]

    df = converter_numericas(df, colunas_numericas)

    for coluna in [
        "instancia",
        "edge_weight_type",
        "algoritmo",
        "estrategia_warmstart",
        "modo",
        "status",
        "motivo_NA",
    ]:
        df[coluna] = df[coluna].astype(str).str.strip()

    df["status_norm"] = normalizar_texto(df["status"])
    df["motivo_norm"] = normalizar_texto(df["motivo_NA"])

    df["algoritmo"] = df["algoritmo"].str.strip()
    df["estrategia_warmstart"] = (
        df["estrategia_warmstart"].str.strip()
    )
    df.loc[
        df["algoritmo"].isin(ALGORITMOS_APROXIMADOS),
        "estrategia_warmstart",
    ] = "NA"

    # A classificação usa motivo_NA como fonte principal.
    # Isso evita o problema de comparar status com a string "NA"
    # depois de o pandas converter essa string para valor ausente.
    df["timeout"] = df["motivo_norm"].eq("timeout")

    df["nao_aplicavel"] = (
        df["motivo_norm"].eq("nao_aplicavel")
    )

    df["concluida"] = (
        df["status_norm"].eq("ok")
        & ~df["timeout"]
        & ~df["nao_aplicavel"]
    )

    # Tempo de busca sem o custo de inicialização da heurística.
    # Só é interpretado quando as duas medidas estão disponíveis.
    if "tempo_heuristica_s" in df.columns:
        df["tempo_busca_s"] = (
            df["tempo_total_s"] - df["tempo_heuristica_s"]
        )

        df.loc[
            df["tempo_busca_s"] < 0,
            "tempo_busca_s"
        ] = np.nan
    else:
        df["tempo_busca_s"] = np.nan

    return df


# ============================================================
# 1. DIAGNÓSTICO GERAL DAS EXECUÇÕES
# ============================================================

def analisar_execucoes(df):
    print("\n" + "=" * 65)
    print("1. DIAGNÓSTICO GERAL DAS EXECUÇÕES")
    print("=" * 65)

    resumo = (
        df.groupby(
            ["algoritmo", "estrategia_warmstart", "modo"],
            dropna=False,
        )
        .agg(
            execucoes=("instancia", "size"),
            concluidas=("concluida", "sum"),
            timeouts=("timeout", "sum"),
            nao_aplicaveis=("nao_aplicavel", "sum"),
        )
        .reset_index()
    )

    salvar_csv(resumo, "01_resumo_execucoes.csv")

    # Contagem de instâncias distintas afetadas por timeout.
    timeouts_instancias = (
        df[df["timeout"]]
        .groupby(
            ["algoritmo", "estrategia_warmstart", "modo"]
        )
        .agg(
            instancias_com_timeout=("instancia", "nunique")
        )
        .reset_index()
    )

    salvar_csv(
        timeouts_instancias,
        "02_instancias_com_timeout.csv",
    )

    # Timeouts preservados por instância; instâncias distintas
    # com o mesmo N não são fundidas.
    timeouts_por_instancia = (
        df[df["timeout"]]
        .groupby(["instancia", "N", "modo"])
        .agg(execucoes_timeout=("instancia", "size"))
        .reset_index()
        .sort_values(["modo", "instancia"])
    )
    salvar_csv(
        timeouts_por_instancia,
        "03_timeouts_por_instancia.csv",
    )

    print("\nResumo de status:")
    print(f"Total de execuções: {len(df)}")
    print(f"Concluídas: {df['concluida'].sum()}")
    print(f"Timeouts: {df['timeout'].sum()}")
    print(f"Não aplicáveis: {df['nao_aplicavel'].sum()}")

    print(
        "\nAtenção: timeout e não aplicável são categorias "
        "diferentes e não devem ser agrupadas."
    )


# ============================================================
# 2. AGREGAÇÃO DAS REPETIÇÕES
# ============================================================

def agregar_execucoes(df):
    """
    Agrega repetições pela mediana.

    Para BnB, mantém cada estratégia de warm-start separada.
    Para CHR/TAT, a estratégia de warm-start pode não se aplicar.
    """

    concluidas = df[df["concluida"]].copy()
    # Memória só pode vir de execuções explicitamente medidas nesse modo.
    concluidas.loc[
        concluidas["modo"] != "memoria", "pico_memoria_MB"
    ] = np.nan

    chaves = [
        "instancia",
        "N",
        "edge_weight_type",
        "algoritmo",
        "estrategia_warmstart",
        "modo",
    ]

    metricas = [
        "custo",
        "gap",
        "tempo_total_s",
        "tempo_busca_s",
        "tempo_heuristica_s",
        "nos_expandidos",
        "nos_podados",
        "profundidade_max",
        "lower_bound_final",
        "melhor_ub_final",
        "pico_memoria_MB",
    ]

    metricas = [
        c for c in metricas
        if c in concluidas.columns
    ]

    agregado = (
        concluidas
        .groupby(chaves, dropna=False)[metricas]
        .median()
        .reset_index()
    )

    repeticoes = (
        concluidas
        .groupby(chaves, dropna=False)
        .size()
        .reset_index(name="n_repeticoes_concluidas")
    )

    agregado = agregado.merge(
        repeticoes,
        on=chaves,
        how="left",
    )

    salvar_csv(
        agregado,
        "04_execucoes_agregadas_mediana.csv",
    )

    return agregado


# ============================================================
# 3. REGRESSÃO DE COMPLEXIDADE EMPÍRICA
# ============================================================

def modelo_linear(x, a, b):
    return a * x + b


def modelo_n2(x, a, b):
    return a * (x ** 2) + b


def modelo_n3(x, a, b):
    return a * (x ** 3) + b


def modelo_nlogn(x, a, b):
    return a * x * np.log(x) + b


def modelo_exponencial(x, a, b):
    return np.exp(a + b * x)


MODELOS = {
    "N": modelo_linear,
    "N²": modelo_n2,
    "N³": modelo_n3,
    "N log N": modelo_nlogn,
}


def ajustar_modelo(x, y, nome_modelo):
    """
    Ajusta uma curva e avalia todas as previsões na escala
    original de y.

    Para o modelo exponencial, o ajuste é feito em log(y),
    mas R² e RMSE são calculados na escala original.
    """

    funcao = MODELOS[nome_modelo]

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    try:
        if nome_modelo == "exponencial":
            if np.any(y <= 0):
                return None

            # log(y) = a + b*N
            b, a = np.polyfit(x, np.log(y), 1)

            params = [a, b]

            with np.errstate(
                over="raise",
                invalid="raise",
            ):
                y_pred = modelo_exponencial(x, a, b)

        else:
            params, _ = curve_fit(
                funcao,
                x,
                y,
                maxfev=20000,
            )

            with np.errstate(
                over="raise",
                invalid="raise",
            ):
                y_pred = funcao(x, *params)

        # Evita aceitar ajustes numericamente inválidos.
        if not np.all(np.isfinite(y_pred)):
            return None

        return {
            "modelo": nome_modelo,
            "parametros": str(
                [float(p) for p in params]
            ),
            "params": [float(p) for p in params],
            "r2": r2_manual(y, y_pred),
            "rmse": rmse_manual(y, y_pred),
            "y_pred": y_pred,
        }

    except (
        RuntimeError,
        ValueError,
        FloatingPointError,
        OverflowError,
        np.linalg.LinAlgError,
    ):
        # Um modelo que não conseguiu ajustar não deve
        # interromper a análise dos demais modelos.
        return None


def regressao_grupo(
    dados,
    coluna_y,
    nome_grupo,
    nome_arquivo,
):
    """
    Ajusta os modelos candidatos para um conjunto de dados.

    Não tenta inferir complexidade assintótica.
    O objetivo é comparar tendências empíricas observadas.
    """

    colunas_dados = ["instancia", "N", coluna_y]
    if "n_repeticoes_concluidas" in dados.columns:
        colunas_dados.append("n_repeticoes_concluidas")
    dados = dados[
        [c for c in colunas_dados if c in dados.columns]
    ].copy()

    dados = dados.dropna()

    dados["N"] = pd.to_numeric(
        dados["N"],
        errors="coerce",
    )

    dados[coluna_y] = pd.to_numeric(
        dados[coluna_y],
        errors="coerce",
    )

    dados = dados.dropna()

    dados = dados[
        (dados["N"] > 0)
        & (dados[coluna_y] > 0)
    ]

    # Cada instância é um ponto independente, mesmo quando N coincide.
    # Ex.: bayg29 e bays29 permanecem como duas observações distintas.
    dados = dados.sort_values([c for c in ["N", "instancia"] if c in dados.columns])

    x = dados["N"].to_numpy(dtype=float)
    y = dados[coluna_y].to_numpy(dtype=float)

    if len(x) < MIN_PONTOS_REGRESSAO:
        print(
            f"[AVISO] {nome_grupo}: apenas {len(x)} instâncias. "
            "Regressão não realizada."
        )
        return None

    resultados = []

    for nome_modelo in MODELOS:

        ajuste = ajustar_modelo(
            x,
            y,
            nome_modelo,
        )

        if ajuste is None:
            print(
                f"[AVISO] Falha no ajuste do modelo "
                f"{nome_modelo} para {nome_grupo}."
            )
            resultados.append({
                "grupo": nome_grupo,
                "metrica": coluna_y,
                "modelo": nome_modelo,
                "parametros": "NA",
                "r2_escala_original": np.nan,
                "rmse_escala_original": np.nan,
                "n_pontos": len(x),
                "n_instancias": len(x),
                "n_repeticoes_mediana_total": int(
                    dados["n_repeticoes_concluidas"].sum()
                ) if "n_repeticoes_concluidas" in dados.columns else np.nan,
            })
            continue

        resultados.append({
            "grupo": nome_grupo,
            "metrica": coluna_y,
            "modelo": ajuste["modelo"],
            "parametros": ajuste["parametros"],
            "r2_escala_original": ajuste["r2"],
            "rmse_escala_original": ajuste["rmse"],
            "n_pontos": len(x),
            "n_instancias": len(x),
            "n_repeticoes_mediana_total": int(
                dados["n_repeticoes_concluidas"].sum()
            ) if "n_repeticoes_concluidas" in dados.columns else np.nan,
        })

    if not resultados:
        return None

    resultados_df = pd.DataFrame(resultados)

    salvar_csv(
        resultados_df,
        nome_arquivo,
    )

    # --------------------------------------------------------
    # Gráfico dos modelos ajustados
    # --------------------------------------------------------

    plt.figure()

    plt.scatter(
        x,
        y,
        color="black",
        label="Dados observados",
        zorder=5,
    )

    x_plot = np.linspace(
        np.min(x),
        np.max(x),
        200,
    )

    for nome_modelo in MODELOS:

        ajuste = ajustar_modelo(
            x,
            y,
            nome_modelo,
        )

        if ajuste is None:
            continue

        funcao = MODELOS[nome_modelo]
        params = ajuste["params"]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                with np.errstate(
                    over="ignore",
                    invalid="ignore",
                ):
                    y_plot = funcao(
                        x_plot,
                        *params,
                    )

            if np.all(np.isfinite(y_plot)):
                plt.plot(
                    x_plot,
                    y_plot,
                    label=nome_modelo,
                )

        except (
            ValueError,
            FloatingPointError,
            OverflowError,
        ):
            continue

    plt.title(
        f"{nome_grupo} — {coluna_y}"
    )

    plt.xlabel("Número de vértices (N)")
    plt.ylabel(coluna_y)
    plt.legend()

    salvar_grafico(
        nome_arquivo.replace(".csv", ".png")
    )

    return resultados_df


def analisar_complexidade(agregado):
    print("\n" + "=" * 65)
    print("2. COMPLEXIDADE EMPÍRICA")
    print("=" * 65)

    resultados = []

    # --------------------------------------------------------
    # BnB: tempo, busca, nós e memória por warm-start
    # --------------------------------------------------------

    bnb_tempo = agregado[
        (agregado["algoritmo"] == "BnB")
        & (agregado["modo"] == "tempo")
    ].copy()

    metricas_bnb = [
        "tempo_total_s",
        "tempo_busca_s",
        "nos_expandidos",
        "nos_podados",
    ]

    for estrategia in ESTRATEGIAS_BNB:
        grupo = bnb_tempo[
            bnb_tempo["estrategia_warmstart"] == estrategia
        ]

        for metrica in metricas_bnb:
            if metrica not in grupo.columns:
                continue

            resultado = regressao_grupo(
                grupo,
                metrica,
                f"BnB_{estrategia}",
                f"regressao_BnB_{estrategia}_{metrica}.csv",
            )

            if resultado is not None:
                resultados.append(resultado)

    bnb_memoria = agregado[
        (agregado["algoritmo"] == "BnB")
        & (agregado["modo"] == "memoria")
    ].copy()

    for estrategia in ESTRATEGIAS_BNB:
        grupo = bnb_memoria[
            bnb_memoria["estrategia_warmstart"] == estrategia
        ]

        if "pico_memoria_MB" in grupo.columns:
            resultado = regressao_grupo(
                grupo,
                "pico_memoria_MB",
                f"BnB_{estrategia}",
                f"regressao_BnB_{estrategia}_memoria.csv",
            )

            if resultado is not None:
                resultados.append(resultado)

    # --------------------------------------------------------
    # CHR e TAT: tempo e memória
    # --------------------------------------------------------

    for algoritmo in ALGORITMOS_APROXIMADOS:

        grupo_tempo = agregado[
            (agregado["algoritmo"] == algoritmo)
            & (agregado["modo"] == "tempo")
        ]

        if not grupo_tempo.empty:
            resultado = regressao_grupo(
                grupo_tempo,
                "tempo_total_s",
                f"{algoritmo} [modo=tempo]",
                f"regressao_{algoritmo}_tempo.csv",
            )

            if resultado is not None:
                resultados.append(resultado)

        grupo_memoria = agregado[
            (agregado["algoritmo"] == algoritmo)
            & (agregado["modo"] == "memoria")
        ]

        if (
            not grupo_memoria.empty
            and "pico_memoria_MB" in grupo_memoria.columns
        ):
            resultado = regressao_grupo(
                grupo_memoria,
                "pico_memoria_MB",
                f"{algoritmo} [modo=memoria]",
                f"regressao_{algoritmo}_memoria.csv",
            )

            if resultado is not None:
                resultados.append(resultado)

    # --------------------------------------------------------
    # Consolidação
    # --------------------------------------------------------

    if resultados:
        salvar_csv(
            pd.concat(resultados, ignore_index=True),
            "05_comparacao_modelos_complexidade.csv",
        )

    print(
        "\nOs ajustes descrevem tendências empíricas das "
        "instâncias observadas. Não demonstram, isoladamente, "
        "a complexidade assintótica dos algoritmos."
    )


# ============================================================
# 4. CUSTO POR NÓ E EFICIÊNCIA DE BUSCA
# ============================================================

def analisar_custo_por_no(agregado):
    print("\n" + "=" * 65)
    print("3. CUSTO POR NÓ E EFICIÊNCIA DE BUSCA")
    print("=" * 65)

    bnb = agregado[
        (agregado["algoritmo"] == "BnB")
        & (agregado["modo"] == "tempo")
    ].copy()

    if bnb.empty:
        print("[AVISO] Não há execuções BnB concluídas no modo tempo.")
        return

    bnb["nos_expandidos"] = pd.to_numeric(
        bnb["nos_expandidos"],
        errors="coerce",
    )

    bnb["tempo_total_s"] = pd.to_numeric(
        bnb["tempo_total_s"],
        errors="coerce",
    )

    bnb["tempo_busca_s"] = pd.to_numeric(
        bnb["tempo_busca_s"],
        errors="coerce",
    )

    bnb["us_por_no_total"] = np.where(
        bnb["nos_expandidos"] > 0,
        1e6 * bnb["tempo_total_s"] / bnb["nos_expandidos"],
        np.nan,
    )

    bnb["us_por_no_busca"] = np.where(
        bnb["nos_expandidos"] > 0,
        1e6 * bnb["tempo_busca_s"] / bnb["nos_expandidos"],
        np.nan,
    )

    # Indicador de poda:
    # só deve ser interpretado se os contadores forem
    # mutuamente exclusivos e cobrirem o mesmo universo de nós.
    bnb["proporcao_poda"] = np.where(
        (bnb["nos_expandidos"] + bnb["nos_podados"]) > 0,
        bnb["nos_podados"] /
        (bnb["nos_expandidos"] + bnb["nos_podados"]),
        np.nan,
    )

    salvar_csv(
        bnb,
        "06_custo_por_no_por_instancia.csv",
    )

    resumo = (
        bnb.groupby("estrategia_warmstart")
        .agg(
            mediana_us_por_no_total=("us_por_no_total", "median"),
            mediana_us_por_no_busca=("us_por_no_busca", "median"),
            mediana_proporcao_poda=("proporcao_poda", "median"),
            instancias=("instancia", "nunique"),
            repeticoes=("n_repeticoes_concluidas", "sum"),
        )
        .reset_index()
    )

    salvar_csv(
        resumo,
        "07_resumo_custo_por_no.csv",
    )
    print("\nResumo de custo por nó (medianas por estratégia):")
    print(resumo.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Gráfico de custo por nó.
    plt.figure()

    for estrategia, grupo in bnb.groupby(
        "estrategia_warmstart"
    ):
        grupo = grupo.sort_values("N")

        plt.plot(
            grupo["N"],
            grupo["us_por_no_busca"],
            marker="o",
            label=estrategia,
        )

    plt.title("Custo médio de busca por nó")
    plt.xlabel("Número de vértices (N)")
    plt.ylabel("Microssegundos por nó expandido")
    plt.legend()

    salvar_grafico("custo_por_no.png")


# ============================================================
# 5. COMPARAÇÃO DOS WARM-STARTS
# ============================================================

def analisar_warmstarts(agregado):
    print("\n" + "=" * 65)
    print("4. COMPARAÇÃO PAREADA DOS WARM-STARTS")
    print("=" * 65)

    bnb_tempo = agregado[
        (agregado["algoritmo"] == "BnB")
        & (agregado["modo"] == "tempo")
    ].copy()
    bnb_memoria = agregado[
        (agregado["algoritmo"] == "BnB")
        & (agregado["modo"] == "memoria")
    ][["instancia", "estrategia_warmstart", "pico_memoria_MB"]].copy()

    if bnb_tempo.empty:
        raise ValueError("Não há execuções BnB concluídas no modo tempo.")

    # A coluna de memória é preenchida exclusivamente com medições do modo memória.
    resumo = bnb_tempo.merge(
        bnb_memoria,
        on=["instancia", "estrategia_warmstart"],
        how="left",
        suffixes=("", "_memoria"),
    )
    if "pico_memoria_MB_memoria" in resumo.columns:
        resumo["pico_memoria_MB"] = resumo["pico_memoria_MB_memoria"]
        resumo.drop(columns=["pico_memoria_MB_memoria"], inplace=True)

    salvar_csv(resumo, "08_warmstarts_por_instancia.csv")
    # Usa o resumo enriquecido com memória medida em modo memória.
    bnb_tempo = resumo.copy()

    comparacoes = []
    estrategias = [e for e in ESTRATEGIAS_BNB if e != "trivial"]
    for estrategia in estrategias:
        for instancia in sorted(set(bnb_tempo["instancia"])):
            par = bnb_tempo[
                bnb_tempo["instancia"].eq(instancia)
                & bnb_tempo["estrategia_warmstart"].isin(["trivial", estrategia])
            ]
            if set(par["estrategia_warmstart"]) != {"trivial", estrategia}:
                continue
            trivial = par[par["estrategia_warmstart"] == "trivial"].iloc[0]
            linha = par[par["estrategia_warmstart"] == estrategia].iloc[0]
            reducao = (
                100 * (trivial["nos_expandidos"] - linha["nos_expandidos"])
                / trivial["nos_expandidos"]
                if pd.notna(trivial["nos_expandidos"])
                and trivial["nos_expandidos"] != 0
                and pd.notna(linha["nos_expandidos"])
                else np.nan
            )
            comparacoes.append({
                "instancia": instancia,
                "N": linha["N"],
                "estrategia": estrategia,
                "nos_expandidos_trivial": trivial["nos_expandidos"],
                "nos_expandidos_estrategia": linha["nos_expandidos"],
                "reducao_nos_percentual": reducao,
                "tempo_total_trivial_s": trivial["tempo_total_s"],
                "tempo_total_estrategia_s": linha["tempo_total_s"],
                "memoria_trivial_MB": trivial["pico_memoria_MB"],
                "memoria_estrategia_MB": linha["pico_memoria_MB"],
                "repeticoes_trivial": trivial["n_repeticoes_concluidas"],
                "repeticoes_estrategia": linha["n_repeticoes_concluidas"],
            })

    comparacoes_df = pd.DataFrame(comparacoes)
    if comparacoes_df.empty:
        raise ValueError("Não há pares BnB/trivial concluídos para comparar.")
    salvar_csv(comparacoes_df, "09_comparacao_warmstarts_vs_trivial.csv")
    for estrategia in estrategias:
        n_estrategia = bnb_tempo[
            bnb_tempo["estrategia_warmstart"].eq(estrategia)
        ]["instancia"].nunique()
        n_trivial = bnb_tempo[
            bnb_tempo["estrategia_warmstart"].eq("trivial")
        ]["instancia"].nunique()
        n_pareadas = comparacoes_df[
            comparacoes_df["estrategia"].eq(estrategia)
        ]["instancia"].nunique()
        print(
            f"{estrategia}: {n_estrategia} instâncias concluídas; "
            f"trivial: {n_trivial}; pares válidos: {n_pareadas}."
        )
        if n_estrategia < n_trivial:
            print(f"  A estratégia {estrategia} concluiu menos instâncias que trivial.")


def analisar_qualidade_aproximacao(agregado):
    print("\n" + "=" * 65)
    print("5. QUALIDADE DAS HEURÍSTICAS DE APROXIMAÇÃO")
    print("=" * 65)

    aproximados = agregado[
        agregado["algoritmo"].isin(
            ALGORITMOS_APROXIMADOS
        )
    ].copy()

    if aproximados.empty:
        print("[AVISO] Não há execuções aproximadas concluídas.")
        return

    # Evita misturar os modos tempo e memória.
    aproximados = aproximados[
        aproximados["modo"] == "tempo"
    ]

    if aproximados.empty:
        print("[AVISO] Não há execuções aproximadas no modo tempo.")
        return

    resumo = (
        aproximados
        .groupby(
            ["instancia", "N", "algoritmo"]
        )
        .agg(
            gap_mediano=("gap", "median"),
            tempo_mediano_s=("tempo_total_s", "median"),
            custo_mediano=("custo", "median"),
            repeticoes=("n_repeticoes_concluidas", "sum"),
        )
        .reset_index()
    )

    salvar_csv(
        resumo,
        "10_qualidade_aproximacao_por_instancia.csv",
    )

    resumo_algoritmo = (
        resumo.groupby("algoritmo")
        .agg(
            gap_mediano=("gap_mediano", "median"),
            gap_maximo=("gap_mediano", "max"),
            gap_minimo=("gap_mediano", "min"),
            tempo_mediano_s=("tempo_mediano_s", "median"),
            instancias=("instancia", "nunique"),
        )
        .reset_index()
    )

    salvar_csv(
        resumo_algoritmo,
        "11_resumo_qualidade_aproximacao.csv",
    )

    # Gráfico de gap por N.
    plt.figure()

    for algoritmo, grupo in resumo.groupby("algoritmo"):
        grupo = grupo.sort_values("N")

        plt.plot(
            grupo["N"],
            grupo["gap_mediano"],
            marker="o",
            label=algoritmo,
        )

    plt.title("Gap das heurísticas de aproximação")
    plt.xlabel("Número de vértices (N)")
    plt.ylabel("Gap")
    plt.legend()

    salvar_grafico("gap_aproximacao.png")

    # Gráfico de tempo por N.
    plt.figure()

    for algoritmo, grupo in resumo.groupby("algoritmo"):
        grupo = grupo.sort_values("N")

        plt.plot(
            grupo["N"],
            grupo["tempo_mediano_s"],
            marker="o",
            label=algoritmo,
        )

    plt.title("Tempo das heurísticas de aproximação")
    plt.xlabel("Número de vértices (N)")
    plt.ylabel("Tempo (s)")
    plt.legend()

    salvar_grafico("tempo_aproximacao.png")


# ============================================================
# 7. VIOLAÇÕES TRIANGULARES
# ============================================================

def analisar_violacoes_triangulares(df):
    print("\n" + "=" * 65)
    print("6. VIOLAÇÕES TRIANGULARES")
    print("=" * 65)

    colunas = [
        "instancia",
        "N",
        "edge_weight_type",
        "violacoes_reais",
        "violacoes_arredondadas",
    ]

    if not all(c in df.columns for c in colunas):
        print("[AVISO] Colunas de violações não encontradas.")
        return

    # Uma linha por instância, evitando contar repetições.
    instancias = (
        df[colunas]
        .drop_duplicates(subset=["instancia"])
        .copy()
    )

    instancias["violacoes_reais"] = pd.to_numeric(
        instancias["violacoes_reais"],
        errors="coerce",
    )

    instancias["violacoes_arredondadas"] = pd.to_numeric(
        instancias["violacoes_arredondadas"],
        errors="coerce",
    )

    # Conforme a implementação da verificação real:
    # EUC_2D e CEIL_2D são aplicáveis.
    aplicaveis = instancias[
        instancias["edge_weight_type"].isin(
            ["EUC_2D", "CEIL_2D"]
        )
        & instancias["violacoes_reais"].notna()
    ].copy()

    n_aplicaveis = len(aplicaveis)

    n_com_violacao_real = (
        aplicaveis["violacoes_reais"] > 0
    ).sum()

    n_com_violacao_arredondada = (
        instancias["violacoes_arredondadas"] > 0
    ).sum()

    resultado = pd.DataFrame([{
        "instancias_total": len(instancias),
        "instancias_aplicaveis_verificacao_real": n_aplicaveis,
        "instancias_com_violacao_real": n_com_violacao_real,
        "instancias_com_violacao_arredondada": (
            n_com_violacao_arredondada
        ),
        "proporcao_violacao_real": (
            n_com_violacao_real / n_aplicaveis
            if n_aplicaveis > 0 else np.nan
        ),
    }])

    salvar_csv(
        resultado,
        "12_resumo_violacoes_triangulares.csv",
    )

    por_tipo = (
        instancias.groupby("edge_weight_type")
        .agg(
            instancias=("instancia", "nunique"),
            violacoes_reais_validas=(
                "violacoes_reais",
                "count",
            ),
            violacoes_reais_positivas=(
                "violacoes_reais",
                lambda s: (s.dropna() > 0).sum(),
            ),
            violacoes_arredondadas_positivas=(
                "violacoes_arredondadas",
                lambda s: (s.dropna() > 0).sum(),
            ),
        )
        .reset_index()
    )

    salvar_csv(
        por_tipo,
        "13_violacoes_por_tipo_distancia.csv",
    )

    print(resultado.to_string(index=False))


# ============================================================
# 8. ANÁLISE DE SENSIBILIDADE DAS FEATURES
# ============================================================

def carregar_features():
    if not ARQUIVO_FEATURES.exists():
        raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {ARQUIVO_FEATURES}")
    features = pd.read_csv(ARQUIVO_FEATURES, keep_default_na=False, na_values=["", "NA"])
    exigir_colunas(
        features,
        ["instancia", "tem_coordenadas", "razao_aspecto",
         "prop_fecho_convexo", "entropia_espacial",
         "densidade_clustering", "grau_max_mst",
         "cv_dist_vizinho_proximo", "dist_media", "dist_desvio"],
        ARQUIVO_FEATURES.name,
    )
    features["instancia"] = features["instancia"].astype(str).str.strip()
    valores = features["tem_coordenadas"].astype(str).str.strip().str.lower()
    mapa = {"true": True, "1": True, "sim": True, "yes": True,
            "false": False, "0": False, "não": False, "nao": False, "no": False}
    features["tem_coordenadas"] = valores.map(mapa)
    if features["tem_coordenadas"].isna().any():
        ruins = features.loc[features["tem_coordenadas"].isna(), "instancia"].tolist()
        raise ValueError(f"tem_coordenadas contém valores inválidos nas instâncias: {ruins}")
    return features

def teste_permutacao_spearman(x, y):
    """
    Calcula Spearman e um p-valor por permutação.

    Para amostras pequenas, não depende da aproximação
    assintótica do p-valor padrão do Spearman.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        return np.nan, np.nan

    resultado = spearmanr(x, y)

    rho = resultado.statistic

    if not np.isfinite(rho):
        return rho, np.nan

    def estatistica(a, b):
        return spearmanr(a, b).statistic

    try:
        permutacao = permutation_test(
            (x, y),
            estatistica,
            permutation_type="pairings",
            alternative="two-sided",
            n_resamples=9999,
            random_state=SEED,
        )

        return rho, permutacao.pvalue

    except Exception:
        return rho, np.nan


def preparar_dados_sensibilidade(features, df):
    """
    Prepara a base para análise de sensibilidade.

    - Calcula a mediana do gap de TAT e CHR usando todas as linhas OK,
      independentemente do modo (tempo ou memória).
    - Junta os gaps às features pela chave instancia.
    - Recupera edge_weight_type de execucoes.csv.
    - Identifica explicitamente instâncias EXPLICIT e instâncias sem
      coordenadas disponíveis.
    """

    # ---------------------------------------------------------
    # 1. Validar colunas necessárias
    # ---------------------------------------------------------
    colunas_execucoes = {
        "instancia",
        "algoritmo",
        "status",
        "gap",
        "edge_weight_type",
    }

    colunas_features = {
        "instancia",
        "tem_coordenadas",
    }

    faltantes_execucoes = colunas_execucoes - set(df.columns)
    faltantes_features = colunas_features - set(features.columns)

    if faltantes_execucoes:
        raise ValueError(
            "Colunas ausentes em execucoes.csv para a sensibilidade: "
            f"{sorted(faltantes_execucoes)}"
        )

    if faltantes_features:
        raise ValueError(
            "Colunas ausentes em features.csv para a sensibilidade: "
            f"{sorted(faltantes_features)}"
        )

    # ---------------------------------------------------------
    # 2. Calcular gap por instância e algoritmo
    #    Usar todas as linhas OK, sem filtrar modo.
    #    Manter TAT e CHR separados.
    # ---------------------------------------------------------

    aproximados_ok = df[
        df["algoritmo"].astype(str).str.strip().str.upper().isin(["TAT", "CHR"])
        & df["status"].astype(str).str.strip().str.upper().eq("OK")
    ].copy()

    aproximados_ok["instancia"] = (
        aproximados_ok["instancia"].astype(str).str.strip()
    )

    aproximados_ok["algoritmo"] = (
        aproximados_ok["algoritmo"].astype(str).str.strip().str.upper()
    )

    aproximados_ok["gap"] = pd.to_numeric(
        aproximados_ok["gap"],
        errors="coerce",
    )

    gaps = (
        aproximados_ok
        .dropna(subset=["gap"])
        .groupby(["instancia", "algoritmo"], as_index=False)
        .agg(gap_mediano=("gap", "median"))
    )

    # Transformar em uma linha por instância e duas colunas:
    # TAT e CHR.
    gaps_instancia = (
        gaps
        .pivot(
            index="instancia",
            columns="algoritmo",
            values="gap_mediano",
        )
        .reset_index()
    )

    gaps_instancia.columns.name = None

    # Garantir que as duas colunas existam, mesmo se algum
    # algoritmo não tiver resultados para determinada amostra.
    for algoritmo in ["TAT", "CHR"]:
        if algoritmo not in gaps_instancia.columns:
            gaps_instancia[algoritmo] = np.nan

    print(
        "\nSensibilidade — instâncias com gap TAT: "
        f"{gaps_instancia['TAT'].notna().sum()}"
    )

    print(
        "Sensibilidade — instâncias com gap CHR: "
        f"{gaps_instancia['CHR'].notna().sum()}"
    )

    # ---------------------------------------------------------
    # 3. Preparar features
    # ---------------------------------------------------------
    features = features.copy()

    # O tipo de distância será recuperado de execucoes.csv.
    # Evita que o merge crie edge_weight_type_x e edge_weight_type_y.
    features = features.drop(
        columns=["edge_weight_type"],
        errors="ignore",
    )

    features["instancia"] = (
        features["instancia"].astype(str).str.strip()
    )

    gaps_instancia["instancia"] = (
        gaps_instancia["instancia"].astype(str).str.strip()
    )

    if features["instancia"].duplicated().any():
        duplicadas = (
            features.loc[
                features["instancia"].duplicated(keep=False),
                "instancia",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "features.csv deve possuir uma linha por instância. "
            f"Instâncias duplicadas: {duplicadas}"
        )

    if gaps_instancia["instancia"].duplicated().any():
        raise ValueError(
            "A agregação dos gaps produziu instâncias duplicadas."
        )

    # ---------------------------------------------------------
    # 4. Recuperar edge_weight_type de execucoes.csv
    # ---------------------------------------------------------
    tipos_distancia = df[
        ["instancia", "edge_weight_type"]
    ].copy()

    tipos_distancia["instancia"] = (
        tipos_distancia["instancia"].astype(str).str.strip()
    )

    tipos_distancia["edge_weight_type"] = (
        tipos_distancia["edge_weight_type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    tipos_distancia = tipos_distancia.dropna(
        subset=["edge_weight_type"]
    )

    conflitos = (
        tipos_distancia
        .groupby("instancia")["edge_weight_type"]
        .nunique()
    )

    conflitos = conflitos[conflitos > 1]

    if not conflitos.empty:
        raise ValueError(
            "Há instâncias com mais de um edge_weight_type "
            "em execucoes.csv: "
            f"{conflitos.index.tolist()}"
        )

    tipos_distancia = (
        tipos_distancia
        .drop_duplicates(subset=["instancia"])
    )

    # ---------------------------------------------------------
    # 5. Juntar features, tipo de distância e gap
    # ---------------------------------------------------------

    features = features.drop(
        columns=["edge_weight_type"],
        errors="ignore",
    )

    dados = features.merge(
        tipos_distancia,
        on="instancia",
        how="left",
        validate="one_to_one",
    )

    dados = dados.merge(
        gaps_instancia,
        on="instancia",
        how="left",
        validate="one_to_one",
    )

    print(
        "Sensibilidade — linhas de features antes da junção: "
        f"{len(features)}"
    )

    print(
        "Sensibilidade — linhas depois da junção: "
        f"{len(dados)}"
    )

    print(
        "Instâncias de features preservadas na junção: "
        f"{dados['instancia'].nunique()} de "
        f"{features['instancia'].nunique()}"
    )

    # ---------------------------------------------------------
    # 6. Identificar instâncias excluídas
    # ---------------------------------------------------------
    dados["edge_weight_type"] = (
        dados["edge_weight_type"]
        .fillna("desconhecido")
        .astype(str)
        .str.upper()
    )

    dados["tem_coordenadas"] = (
        dados["tem_coordenadas"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "sim", "yes"])
    )

    explicitas = dados[
        dados["edge_weight_type"].eq("EXPLICIT")
    ]

    sem_coordenadas = dados[
        (~dados["tem_coordenadas"])
        & (~dados["edge_weight_type"].eq("EXPLICIT"))
    ]

    print(
        "\nInstâncias excluídas por tem_coordenadas=False:"
    )

    for _, linha in dados[~dados["tem_coordenadas"]].iterrows():
        if linha["edge_weight_type"] == "EXPLICIT":
            motivo = "matriz EXPLICIT, sem coordenadas"
        elif linha["edge_weight_type"] == "DESCONHECIDO":
            motivo = "tipo de distância não identificado em execucoes.csv"
        else:
            motivo = "tem_coordenadas=False"

        print(f"  {linha['instancia']}: {motivo}")

    # ---------------------------------------------------------
    # 7. Listar GEO e ATT
    # ---------------------------------------------------------
    geo_att = dados[
        dados["edge_weight_type"].isin(["GEO", "ATT"])
    ]

    print(
        "\nInstâncias GEO/ATT — features calculadas sobre "
        "coordenadas não euclidianas:"
    )

    if geo_att.empty:
        print("  Nenhuma.")
    else:
        for _, linha in geo_att.iterrows():
            print(
                f"  {linha['instancia']}: "
                f"{linha['edge_weight_type']}"
            )

    # ---------------------------------------------------------
    # 8. Retornar somente instâncias com coordenadas
    #    e pelo menos um gap disponível.
    # ---------------------------------------------------------

    dados_validos = dados[
        dados["tem_coordenadas"]
    ].copy()

    dados_validos = dados_validos.dropna(
        subset=["TAT", "CHR"],
        how="all",
    )

    if dados_validos.empty:
        raise ValueError(
            "Nenhuma instância com coordenadas e gap disponível "
            "para TAT ou CHR."
        )

    return dados_validos

def analisar_correlacoes_features(dados):

    print("\n" + "=" * 65)
    print("7. ANÁLISE DE SENSIBILIDADE — CORRELAÇÕES SPEARMAN")
    print("=" * 65)

    features_numericas = [
        "razao_aspecto",
        "prop_fecho_convexo",
        "entropia_espacial",
        "densidade_clustering",
        "grau_max_mst",
        "cv_dist_vizinho_proximo",
        "dist_media",
        "dist_desvio",
    ]

    resultados = []

    for algoritmo in ["TAT", "CHR"]:

        if algoritmo not in dados.columns:
            raise ValueError(
                f"Coluna de gap ausente: {algoritmo}"
            )

        print(
            f"\n{algoritmo} — correlação entre features e gap"
        )

        for feature in features_numericas:

            amostra = dados[
                ["instancia", feature, algoritmo]
            ].copy()

            amostra[feature] = pd.to_numeric(
                amostra[feature],
                errors="coerce",
            )

            amostra[algoritmo] = pd.to_numeric(
                amostra[algoritmo],
                errors="coerce",
            )

            amostra = amostra.dropna(
                subset=[feature, algoritmo]
            )

            n = amostra["instancia"].nunique()

            observacao = ""

            if n < 4:
                rho = np.nan
                p_valor = np.nan
                observacao = "n insuficiente"

            else:
                x = amostra[feature].to_numpy(dtype=float)
                y = amostra[algoritmo].to_numpy(dtype=float)

                rho, p_valor = teste_permutacao_spearman(
                    x,
                    y,
                )

                if amostra[feature].duplicated().any():
                    observacao = "empates na feature"

                elif amostra[algoritmo].duplicated().any():
                    observacao = "empates no gap"

            resultados.append({
                "algoritmo": algoritmo,
                "feature": feature,
                "n": n,
                "spearman_rho": rho,
                "p_valor": p_valor,
                "observacao": observacao,
            })

            print(
                f"  {feature}: "
                f"rho={rho:.4f}, "
                f"p={p_valor:.4f}, "
                f"n={n} "
                f"({observacao})"
                if np.isfinite(rho)
                else
                f"  {feature}: n={n} — {observacao}"
            )

    resultados_df = pd.DataFrame(resultados)

    salvar_csv(
        resultados_df,
        "14_spearman_features_gap.csv",
    )

    print(
        "\nInterpretação: correlações são indícios de associação, "
        "não evidência de causalidade."
    )

# ============================================================
# 9. MODELO MULTIVARIADO EXPLORATÓRIO
# ============================================================

def analisar_modelo_multivariado(dados):
    """
    Usa uma árvore de decisão rasa para explorar relações
    entre features e gap.

    Com poucos dados, não força o ajuste: a análise é
    ignorada até haver amostras suficientes.

    A árvore não deve ser interpretada como modelo preditivo
    validado quando a amostra é pequena.
    """

    print("\n" + "=" * 65)
    print("8. MODELO MULTIVARIADO EXPLORATÓRIO")
    print("=" * 65)

    if dados is None or dados.empty:
        print("[AVISO] Dados de features indisponíveis.")
        return

    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    colunas_excluidas = {
        "instancia",
        "N",
        "CHR",
        "TAT",
    }

    features_numericas = [
        c for c in dados.select_dtypes(
            include=np.number
        ).columns
        if c not in colunas_excluidas
    ]

    if not features_numericas:
        print("[AVISO] Nenhuma feature numérica.")
        return

    resultados = []

    for algoritmo in ALGORITMOS_APROXIMADOS:
        if algoritmo not in dados.columns:
            continue

        amostra = dados[
            features_numericas + [algoritmo]
        ].copy()

        amostra[algoritmo] = pd.to_numeric(
            amostra[algoritmo],
            errors="coerce",
        )

        amostra = amostra.dropna(
            subset=[algoritmo]
        )

        X = amostra[features_numericas]
        y = amostra[algoritmo]

        n = len(amostra)
        p = len(features_numericas)

        if n < max(10, MIN_AMOSTRAS_POR_FEATURE * p):
            print(
                f"[AVISO] {algoritmo}: {n} amostras para "
                f"{p} features. Modelo multivariado não ajustado."
            )
            continue

        pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "arvore",
                DecisionTreeRegressor(
                    max_depth=2,
                    min_samples_leaf=3,
                    random_state=SEED,
                ),
            ),
        ])

        n_splits = min(5, n)

        cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=SEED,
        )

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="r2",
        )

        pipeline.fit(X, y)

        arvore = pipeline.named_steps["arvore"]

        importancias = pd.DataFrame({
            "algoritmo": algoritmo,
            "feature": features_numericas,
            "importancia": arvore.feature_importances_,
        }).sort_values(
            "importancia",
            ascending=False,
        )

        salvar_csv(
            importancias,
            f"importancia_features_{algoritmo}.csv",
        )

        resultados.append({
            "algoritmo": algoritmo,
            "n": n,
            "n_features": p,
            "r2_cv_mediano": np.median(scores),
            "r2_cv_media": np.mean(scores),
            "r2_cv_desvio": np.std(scores),
        })

    if resultados:
        salvar_csv(
            pd.DataFrame(resultados),
            "16_resultado_modelo_multivariado.csv",
        )


# ============================================================
# 10. PROFILING DE SUBROTINAS
# ============================================================

def analisar_profiling(agregado):
    if not ARQUIVO_PROFILING.exists():
        raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {ARQUIVO_PROFILING}")
    print("\n" + "=" * 65)
    print("9. PROFILING DE SUBROTINAS")
    print("=" * 65)
    profiling = pd.read_csv(ARQUIVO_PROFILING, keep_default_na=False, na_values=["", "NA"])
    exigir_colunas(profiling, ["instancia", "algoritmo", "estrategia", "modo", "repeticao", "subrotina", "chamadas", "tempo_s", "memoria_pico_MB"], ARQUIVO_PROFILING.name)
    profiling = converter_numericas(profiling, ["repeticao", "chamadas", "tempo_s", "memoria_pico_MB"])
    profiling["instancia"] = profiling["instancia"].astype(str).str.strip()
    profiling["algoritmo"] = profiling["algoritmo"].astype(str).str.strip()
    profiling["modo"] = profiling["modo"].astype(str).str.strip().str.lower()
    profiling["estrategia"] = profiling["estrategia"].astype(str).str.strip()
    profiling.loc[profiling["algoritmo"].isin(ALGORITMOS_APROXIMADOS), "estrategia"] = "NA"
    profiling["subrotina"] = profiling["subrotina"].astype(str).str.strip()
    profiling = profiling[profiling["modo"] == "tempo"].copy()
    chaves = ["instancia", "algoritmo", "estrategia", "subrotina"]
    subrotinas_aprox = ["grau_mst", "prim_mst", "emparelhamento", "euler", "atalhos", "validar_e_custo"]
    subrotinas_bnb = ["matriz_reducao", "calculo_lb", "expansao_no", "busca_total", "warmstart", "busca_total_externa"]
    profiling = profiling[
        ((profiling["algoritmo"].isin(ALGORITMOS_APROXIMADOS)) & profiling["subrotina"].isin(subrotinas_aprox))
        | ((profiling["algoritmo"] == "BnB") & profiling["subrotina"].isin(subrotinas_bnb))
    ].copy()
    if profiling.empty:
        raise ValueError("profiling.csv não contém as subrotinas esperadas no modo tempo.")
    med = profiling.groupby(chaves, dropna=False).agg(
        tempo_subrotina_mediano_s=("tempo_s", "median"),
        chamadas_mediana=("chamadas", "median"),
        n_repeticoes=("repeticao", "nunique"),
    ).reset_index()
    exec_tempo = agregado[agregado["modo"] == "tempo"].copy()
    # Tempo total CHR/TAT por instância; BnB por estratégia.
    exec_tempo = exec_tempo[["instancia", "algoritmo", "estrategia_warmstart", "tempo_total_s"]]
    med = med.merge(exec_tempo, left_on=["instancia", "algoritmo", "estrategia"],
                    right_on=["instancia", "algoritmo", "estrategia_warmstart"], how="left")
    med.drop(columns=["estrategia_warmstart"], inplace=True, errors="ignore")
    med.loc[med["algoritmo"].isin(ALGORITMOS_APROXIMADOS), "estrategia"] = "NA"
    med["denominador_s"] = np.where(
        med["algoritmo"].isin(ALGORITMOS_APROXIMADOS), med["tempo_total_s"], np.nan
    )
    # CHR/TAT: subrotinas não aninhadas, participação no tempo total do algoritmo.
    mask_aprox = med["algoritmo"].isin(ALGORITMOS_APROXIMADOS)
    med.loc[mask_aprox, "denominador_s"] = med.loc[mask_aprox, "tempo_total_s"]
    # BnB: denominador interno busca_total; não somar subrotinas aninhadas.
    for instancia in med.loc[med["algoritmo"] == "BnB", "instancia"].unique():
        grupo = med[(med["algoritmo"] == "BnB") & (med["instancia"] == instancia)]
        busca = grupo.loc[grupo["subrotina"] == "busca_total", "tempo_subrotina_mediano_s"]
        if not busca.empty:
            med.loc[grupo.index, "denominador_s"] = busca.iloc[0]
    med["fracao_tempo"] = med["tempo_subrotina_mediano_s"] / med["denominador_s"]
    med["fracao_tempo_percentual"] = 100 * med["fracao_tempo"]
    salvar_csv(med, "18_participacao_tempo_profiling.csv")
    for algoritmo in ["CHR", "TAT"]:
        grupo = med[med["algoritmo"] == algoritmo]
        if grupo.empty:
            raise ValueError(f"profiling.csv não contém registros de {algoritmo} no modo tempo.")
        pivot = grupo.pivot_table(index="instancia", columns="subrotina", values="fracao_tempo_percentual", aggfunc="first")
        col = "emparelhamento" if algoritmo == "CHR" else "emparelhamento"
        print(f"\n{algoritmo} — frações por instância (% do tempo total):")
        print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))
        if col in pivot.columns:
            print(f"Fração de emparelhamento por instância ({algoritmo}):")
            print(pivot[col].to_string(float_format=lambda x: f"{x:.2f}%"))
        soma = pivot.sum(axis=1, min_count=1)
        if (soma.dropna() > 100.0001).any():
            raise ValueError(f"Soma de subrotinas CHR/TAT acima de 100% para {algoritmo}; verificar denominador/duplicatas.")
    bnb = med[med["algoritmo"] == "BnB"]
    if not bnb.empty:
        print("\nBnB — fração de cada subrotina em relação a busca_total (%; subrotinas aninhadas, não somar):")
        print(bnb[["instancia", "estrategia", "subrotina", "fracao_tempo_percentual"]].to_string(index=False))
        print("busca_total_externa inclui overhead do processo.")

def gerar_relatorio_limitacoes(df, agregado):
    linhas = []

    instancias = (
        df[["instancia", "N"]]
        .drop_duplicates()
        .sort_values("N")
    )

    linhas.append(
        f"Quantidade de instâncias: {len(instancias)}"
    )

    linhas.append(
        f"Quantidade de valores distintos de N: "
        f"{instancias['N'].nunique()}"
    )

    linhas.append(
        f"Quantidade de execuções: {len(df)}"
    )

    linhas.append(
        f"Execuções concluídas: {df['concluida'].sum()}"
    )

    linhas.append(
        f"Execuções com timeout: {df['timeout'].sum()}"
    )

    linhas.append(
        f"Execuções não aplicáveis: {df['nao_aplicavel'].sum()}"
    )

    linhas.append("")
    linhas.append("LIMITAÇÕES IMPORTANTES")
    linhas.append(
        "- Os resultados representam somente as instâncias testadas."
    )
    linhas.append(
        "- Timeouts causam censura: as instâncias mais difíceis "
        "podem estar sub-representadas nas regressões de tempo."
    )
    linhas.append(
        "- Não se deve interpretar R² elevado como prova "
        "de complexidade assintótica."
    )
    linhas.append(
        "- Instâncias TSPLIB diferentes podem ter estruturas "
        "distintas mesmo quando possuem o mesmo N."
    )
    linhas.append(
        "- Repetições da mesma instância não equivalem a novas "
        "instâncias independentes."
    )
    linhas.append(
        "- A análise multivariada de features depende da quantidade "
        "de instâncias distintas, não do número de execuções."
    )
    linhas.append(
        "- A memória medida por tracemalloc representa alocações "
        "Python e não necessariamente o RSS total do processo."
    )

    caminho = PASTA_SAIDA / "19_limitacoes_amostra.txt"

    caminho.write_text(
        "\n".join(linhas),
        encoding="utf-8",
    )

    print(f"[OK] {caminho.name}")



# ============================================================
# 12. RELATÓRIO CONSOLIDADO NO TERMINAL
# ============================================================

def _ler_resultado_csv(nome):
    """Lê um resultado auxiliar, se ele tiver sido gerado."""
    caminho = PASTA_SAIDA / nome

    if not caminho.exists():
        return None

    try:
        return pd.read_csv(caminho, keep_default_na=False, na_values=[""])
    except Exception:
        return None


def _imprimir_tabela(titulo, dados, colunas=None, max_linhas=12):
    print(f"\n{titulo}")
    print("-" * 65)

    if dados is None or dados.empty:
        print("Dados indisponíveis ou insuficientes.")
        return

    tabela = dados.copy()

    if colunas is not None:
        colunas = [c for c in colunas if c in tabela.columns]
        tabela = tabela[colunas]

    # Limita a largura e a quantidade de linhas para manter
    # o relatório legível em terminais convencionais.
    if len(tabela) > max_linhas:
        tabela = tabela.head(max_linhas)
        mostrar_aviso = True
    else:
        mostrar_aviso = False

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 150,
        "display.max_colwidth", 28,
        "display.float_format", lambda x: f"{x:.4f}",
    ):
        print(tabela.to_string(index=False))

    if mostrar_aviso:
        print(f"... exibidas {max_linhas} de {len(dados)} linhas.")


def imprimir_relatorio_geral(df, agregado, ambiente):
    print("\n\n" + "=" * 78)
    print("                    RELATÓRIO GERAL DOS EXPERIMENTOS")
    print("                         TSP / BRANCH-AND-BOUND")
    print("=" * 78)

    print("\n[0] AMBIENTE DE EXECUÇÃO")

    _imprimir_tabela(
        "Configuração da máquina utilizada nos experimentos",
        ambiente,
        max_linhas=100,
    )

    salvar_csv(
        ambiente,
        "00_ambiente_execucao.csv",
    )

    instancias = df[["instancia", "N"]].drop_duplicates()
    n_instancias = instancias["instancia"].nunique()
    n_tamanhos = instancias["N"].nunique()
    concluidas, timeouts, nao_aplicaveis, total = int(df["concluida"].sum()), int(df["timeout"].sum()), int(df["nao_aplicavel"].sum()), len(df)
    print("\n[1] VISÃO GERAL")
    print(f"Instâncias: {n_instancias}; tamanhos N: {n_tamanhos}; execuções: {total}; OK: {concluidas}; timeouts: {timeouts}; não aplicáveis: {nao_aplicaveis}")
    distribuicao = df.groupby(["algoritmo", "modo"], dropna=False).agg(exec=("instancia", "size"), OK=("concluida", "sum"), timeouts=("timeout", "sum")).reset_index()
    _imprimir_tabela("[2] EXECUÇÕES POR ALGORITMO E MODO", distribuicao)

    print("\n[3] CUSTO POR NÓ E EFICIÊNCIA DE BUSCA")
    custo_no = _ler_resultado_csv("07_resumo_custo_por_no.csv")
    if custo_no is None or custo_no.empty:
        raise ValueError("Resumo de custo por nó não foi gerado; verifique a seção 3.")
    _imprimir_tabela("Medianas por estratégia", custo_no, max_linhas=1000)

    print("\n[4] WARM-START PAREADO — somente pares OK")
    comp = _ler_resultado_csv("09_comparacao_warmstarts_vs_trivial.csv")
    if comp is None or comp.empty:
        raise ValueError("Comparação pareada de warm-starts não foi gerada.")
    _imprimir_tabela("Comparação por instância", comp, ["instancia", "N", "estrategia", "nos_expandidos_trivial", "nos_expandidos_estrategia", "reducao_nos_percentual", "tempo_total_trivial_s", "tempo_total_estrategia_s", "memoria_trivial_MB", "memoria_estrategia_MB", "repeticoes_trivial", "repeticoes_estrategia"], max_linhas=1000)
    mem_timeout = df[(df["algoritmo"] == "BnB") & (df["modo"] == "memoria") & df["timeout"]].groupby("estrategia_warmstart").agg(timeouts_memoria=("instancia", "size"), instancias=("instancia", "nunique")).reset_index()
    _imprimir_tabela("Timeouts BnB no modo memória por estratégia", mem_timeout)

    print("\n[5] HEURÍSTICAS APROXIMATIVAS")
    aprox = agregado[(agregado["algoritmo"].isin(ALGORITMOS_APROXIMADOS)) & (agregado["modo"] == "tempo")]
    aprox = aprox.groupby(["instancia", "N", "algoritmo"], dropna=False).agg(gap_mediano=("gap", "median"), tempo_mediano_s=("tempo_total_s", "median"), repeticoes=("n_repeticoes_concluidas", "sum")).reset_index()
    _imprimir_tabela("Medianas por instância (com número de repetições)", aprox, max_linhas=1000)

    print("\n[6] MODELOS DE REGRESSÃO EMPÍRICA — todos os modelos")
    complexidade = _ler_resultado_csv("05_comparacao_modelos_complexidade.csv")
    if complexidade is None or complexidade.empty:
        raise ValueError("Não há resultados de regressão empírica para apresentar.")
    _imprimir_tabela("R² e RMSE por série/modelo", complexidade, ["grupo", "metrica", "modelo", "r2_escala_original", "rmse_escala_original", "n_pontos", "n_repeticoes_mediana_total"], max_linhas=1000)
    print("Séries TAT/CHR identificadas como [modo=tempo] usam somente as instâncias do modo tempo; [modo=memoria] usam somente as do modo memória. n_pontos é o número de instâncias e n_repeticoes_mediana_total é a soma das repetições que compõem as medianas dessas instâncias.")
    validos = complexidade["r2_escala_original"].dropna()
    print("Nenhum modelo com R² alto deve ser interpretado como classe de complexidade algorítmica.")
    if not validos.empty and (validos < 0.5).all():
        print("Todos os R² calculados são baixos (abaixo de 0,5) nesta amostra.")

    print("\n[7] VERIFICAÇÃO TRIANGULAR")
    _imprimir_tabela("Resumo", _ler_resultado_csv("12_resumo_violacoes_triangulares.csv"))
    print("EXPLICIT: a não metricidade decorre da própria matriz de custos; violacoes_reais é NA e não se aplica.")
    print("EUC_2D e GEO/ATT: a não metricidade observada pode decorrer apenas do arredondamento das distâncias.")

    print("\n[8] SENSIBILIDADE — SPEARMAN")
    corr = _ler_resultado_csv("14_spearman_features_gap.csv")
    if corr is None or corr.empty:
        raise ValueError("Resultados de sensibilidade não foram gerados.")
    _imprimir_tabela("Correlação por algoritmo e feature", corr, ["algoritmo", "feature", "n", "spearman_rho", "p_valor", "observacao"], max_linhas=1000)
    print("Correlação é indício de associação, não causalidade; o n de instâncias limita a inferência.")

    print("\n[9] PROFILING")
    prof = _ler_resultado_csv("18_participacao_tempo_profiling.csv")
    if prof is None or prof.empty:
        raise ValueError("Resumo de profiling não foi gerado.")
    _imprimir_tabela("Frações por instância/subrotina", prof, ["instancia", "algoritmo", "estrategia", "subrotina", "tempo_subrotina_mediano_s", "denominador_s", "fracao_tempo_percentual", "n_repeticoes"], max_linhas=1000)
    print("CHR/TAT: subrotinas não aninhadas e somáveis. BnB: subrotinas aninhadas; não somar. busca_total_externa inclui overhead do processo.")

    print("\n[10] LIMITAÇÕES CONHECIDAS")
    print(f"- Amostra limitada a {n_instancias} instâncias; instâncias com mesmo N permanecem pontos distintos.")
    print("- Repetições medem variabilidade experimental, mas não aumentam o número de instâncias independentes.")
    print("- Correlações Spearman são exploratórias e não implicam causalidade.")
    print("- Ajustes empíricos não demonstram complexidade assintótica.")
    print("- GEO/ATT usam coordenadas não euclidianas para as features; a interpretação geométrica deve ser cautelosa.")
    print("\n" + "=" * 78 + "\nFIM DO RELATÓRIO GERAL\n" + "=" * 78)

def executar_analises():
    print("=" * 65)
    print("ANÁLISE EXPERIMENTAL — TSP / BRANCH-AND-BOUND")
    print("=" * 65)

    if not ARQUIVO_AMBIENTE.exists():
        raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {ARQUIVO_AMBIENTE}")
    ambiente = pd.read_csv(ARQUIVO_AMBIENTE, keep_default_na=False, na_values=["", "NA"])
    if ambiente.empty:
        raise ValueError(f"O arquivo {ARQUIVO_AMBIENTE.name} está vazio.")
    df = carregar_execucoes()

    analisar_execucoes(df)

    agregado = agregar_execucoes(df)

    analisar_complexidade(agregado)

    analisar_custo_por_no(agregado)

    analisar_warmstarts(agregado)

    analisar_qualidade_aproximacao(agregado)

    analisar_violacoes_triangulares(df)

    features = carregar_features()

    if features is not None:
        dados_sensibilidade = preparar_dados_sensibilidade(
            features,
            df,
        )

        analisar_correlacoes_features(
            dados_sensibilidade
        )


    analisar_profiling(agregado)

    gerar_relatorio_limitacoes(
        df,
        agregado,
    )

    imprimir_relatorio_geral(
        df,
        agregado,
        ambiente,
    )

    print("\n" + "=" * 65)
    print("ANÁLISES FINALIZADAS")
    print(f"Resultados salvos em: {PASTA_SAIDA}")
    print("=" * 65)


class EscritaMultipla:
    def __init__(self, *destinos):
        self.destinos = destinos

    def write(self, texto):
        for destino in self.destinos:
            destino.write(texto)
            destino.flush()

    def flush(self):
        for destino in self.destinos:
            destino.flush()


def main():
    ARQUIVO_RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    with open(ARQUIVO_RELATORIO, "w", encoding="utf-8") as arquivo:
        saida = EscritaMultipla(sys.stdout, arquivo)
        with redirect_stdout(saida):
            executar_analises()
    print(f"Relatório textual salvo em: {ARQUIVO_RELATORIO}")


if __name__ == "__main__":
    main()