
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import networkx as nx


RAIZ = Path(__file__).resolve().parent
NA = "NA"

CAMPOS_EXECUCOES = [
    "instancia", "N", "grupo_N", "edge_weight_type",
    "tem_coordenadas", "otimo", "otimo_fonte",
    "desig_triangular", "violacoes_reais",
    "violacoes_arredondadas", "algoritmo",
    "estrategia_warmstart", "modo", "repeticao",
    "custo", "gap", "tempo_total_s",
    "tempo_heuristica_s", "nos_expandidos",
    "nos_podados", "profundidade_max",
    "lower_bound_final", "melhor_ub_final",
    "pico_memoria_MB", "status", "motivo_NA",
    "ambiente_id",
]

CAMPOS_PROFILING = [
    "instancia", "algoritmo", "estrategia",
    "modo", "repeticao", "subrotina",
    "chamadas", "tempo_s", "memoria_pico_MB",
]

CAMPOS_AMBIENTE = [
    "ambiente_id", "data", "sistema_operacional",
    "cpu", "nucleos_fisicos", "nucleos_logicos",
    "ram_GB", "python_versao", "numpy_versao",
    "networkx_versao", "modo", "workers",
]

# As sete combinações experimentais obrigatórias.
COMBINACOES = (
    ("BnB", "trivial"),
    ("BnB", "vizinho"),
    ("BnB", "fecho"),
    ("BnB", "TAT"),
    ("BnB", "CHR"),
    ("TAT", "NA"),
    ("CHR", "NA"),
)

MOTIVOS_REEXECUTAVEIS = {"erro", "timeout_externo"}


# ---------------------------------------------------------------------
# Configuração e geração de tarefas
# ---------------------------------------------------------------------

def ler_config(caminho):
    """Carrega os metadados das instâncias experimentais."""
    with open(caminho, encoding="utf-8") as arquivo:
        return json.load(arquivo)


def gerar_tarefas(instancias, reps, modo):
    """
    Gera as sete combinações para cada instância e repetição.

    Retorna tuplas:
        (instancia, algoritmo, estrategia, modo, repeticao)
    """
    if reps < 1:
        raise ValueError("reps deve ser pelo menos 1.")

    tarefas = []

    instancias_ordenadas = sorted(
        instancias,
        key=lambda inst: int(inst["N"])
    )

    for inst in instancias_ordenadas:
        for repeticao in range(1, reps + 1):
            for algoritmo, estrategia in COMBINACOES:
                tarefas.append(
                    (inst, algoritmo, estrategia, modo, repeticao)
                )

    return tarefas


def tarefa_para_dict(tarefa):
    """Converte uma tupla de tarefa em um dicionário nomeado."""
    instancia, algoritmo, estrategia, modo, repeticao = tarefa

    return {
        "instancia": instancia,
        "algoritmo": algoritmo,
        "estrategia": estrategia,
        "modo": modo,
        "repeticao": repeticao,
    }


def chave_tarefa(tarefa):
    """Retorna a chave única de uma tarefa experimental."""
    if not isinstance(tarefa, dict):
        tarefa = tarefa_para_dict(tarefa)

    return (
        tarefa["instancia"]["nome"],
        tarefa["algoritmo"],
        tarefa["estrategia"],
        tarefa["modo"],
        str(tarefa["repeticao"]),
    )


# ---------------------------------------------------------------------
# CSV e retomada
# ---------------------------------------------------------------------

def garantir_cabecalho(caminho, campos):
    """Cria o CSV e seu cabeçalho quando necessário."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    if not caminho.exists() or caminho.stat().st_size == 0:
        with caminho.open("w", newline="", encoding="utf-8") as arquivo:
            writer = csv.DictWriter(arquivo, fieldnames=campos)
            writer.writeheader()


def append_csv(caminho, campos, linha):
    """Acrescenta uma linha ao CSV e força sua gravação em disco."""
    caminho = Path(caminho)
    garantir_cabecalho(caminho, campos)

    with caminho.open("a", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=campos)
        writer.writerow(linha)
        arquivo.flush()
        os.fsync(arquivo.fileno())


def carregar_chaves(caminho):
    """
    Carrega as tarefas concluídas e remove do CSV as execuções
    que devem ser reexecutadas.

    Chave:
        (instancia, algoritmo, estrategia, modo, repeticao)

    Para BnB, a estratégia está em estrategia_warmstart.
    Para TAT/CHR, utiliza-se NA.
    """

    caminho = Path(caminho)

    if not caminho.exists():
        return set()

    with caminho.open("r", newline="", encoding="utf-8-sig") as f:
        leitor = csv.DictReader(f)
        linhas = list(leitor)

    chaves_concluidas = set()
    linhas_validas = []

    motivos_reexecutaveis = {
        "erro",
        "timeout_externo"
    }

    for linha in linhas:
        motivo = linha.get("motivo_NA", "")
        status = linha.get("status", "")

        falhou = (
            status == "NA"
            and motivo in motivos_reexecutaveis
        )

        if falhou:
            continue

        linhas_validas.append(linha)

        algoritmo = linha["algoritmo"]

        estrategia = (
            linha["estrategia_warmstart"]
            if algoritmo == "BnB"
            else "NA"
        )

        chave = (
            linha["instancia"],
            algoritmo,
            estrategia,
            linha["modo"],
            str(linha["repeticao"])
        )

        chaves_concluidas.add(chave)

    # Reescreve o CSV sem as execuções que serão refeitas.
    with caminho.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(
            f,
            fieldnames=CAMPOS_EXECUCOES
        )

        escritor.writeheader()
        escritor.writerows(linhas_validas)

    return chaves_concluidas


def obter_resumo_instancias(tarefas, concluidas):
    """Calcula total, concluídas e pendentes por instância."""
    resumo = {}

    for tarefa in tarefas:
        inst, algoritmo, estrategia, modo, repeticao = tarefa
        nome = inst["nome"]

        if nome not in resumo:
            resumo[nome] = {
                "N": inst["N"],
                "total": 0,
                "concluidas": 0,
            }

        resumo[nome]["total"] += 1

        if chave_tarefa(tarefa) in concluidas:
            resumo[nome]["concluidas"] += 1

    for dados in resumo.values():
        dados["pendentes"] = dados["total"] - dados["concluidas"]

    return resumo


def validar_quantidade_tarefas(tarefas, reps):
    """Aborta se alguma instância não tiver exatamente sete tarefas por repetição."""
    esperado = 7 * reps
    contagem = {}

    for tarefa in tarefas:
        nome = tarefa[0]["nome"]
        contagem[nome] = contagem.get(nome, 0) + 1

    for nome, quantidade in contagem.items():
        if quantidade != esperado:
            raise RuntimeError(
                f"Instância {nome}: esperadas {esperado} tarefas, "
                f"mas foram geradas {quantidade}."
            )


def imprimir_resumo(tarefas, concluidas, modo, reps):
    """Imprime o total por modo e a situação de cada instância."""
    resumo = obter_resumo_instancias(tarefas, concluidas)

    total = len(tarefas)
    ja_concluidas = sum(
        dados["concluidas"] for dados in resumo.values()
    )
    pendentes = total - ja_concluidas

    print(f"\nModo: {modo}")
    print(f"Total de tarefas: {total}")
    print(f"Já concluídas: {ja_concluidas}")
    print(f"Pendentes: {pendentes}")
    print("\nDesdobramento por instância:")

    print(
        f"{'Instância':<18}"
        f"{'N':>6}"
        f"{'Total':>10}"
        f"{'Concluídas':>14}"
        f"{'Pendentes':>12}"
    )

    for nome, dados in resumo.items():
        print(
            f"{nome:<18}"
            f"{dados['N']:>6}"
            f"{dados['total']:>10}"
            f"{dados['concluidas']:>14}"
            f"{dados['pendentes']:>12}"
        )

    esperado = 7 * reps

    if any(dados["total"] != esperado for dados in resumo.values()):
        raise RuntimeError(
            f"Quantidade incorreta de tarefas por instância. "
            f"Esperado: {esperado}."
        )


# ---------------------------------------------------------------------
# Ambiente computacional
# ---------------------------------------------------------------------

def coletar_ambiente(modo, workers):
    """Coleta informações sobre o ambiente computacional."""
    try:
        import psutil

        nucleos_fisicos = psutil.cpu_count(logical=False) or NA
        nucleos_logicos = psutil.cpu_count(logical=True) or NA
        ram = psutil.virtual_memory().total / (1024 ** 3)

    except ImportError:
        nucleos_fisicos = NA
        nucleos_logicos = os.cpu_count() or NA
        ram = NA

    return {
        "ambiente_id": str(uuid.uuid4()),
        "data": datetime.now().astimezone().isoformat(),
        "sistema_operacional": platform.platform(),
        "cpu": platform.processor() or NA,
        "nucleos_fisicos": nucleos_fisicos,
        "nucleos_logicos": nucleos_logicos,
        "ram_GB": ram,
        "python_versao": platform.python_version(),
        "numpy_versao": np.__version__,
        "networkx_versao": nx.__version__,
        "modo": modo,
        "workers": workers,
    }


def obter_ou_gravar_ambiente(caminho, ambiente):
    """Reutiliza o ambiente registrado ou grava um novo."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    if caminho.exists() and caminho.stat().st_size > 0:
        with caminho.open(
            newline="", encoding="utf-8-sig"
        ) as arquivo:
            leitor = csv.DictReader(arquivo)

            if leitor.fieldnames != CAMPOS_AMBIENTE:
                raise ValueError(
                    f"Cabeçalho incorreto em {caminho}. "
                    "Verifique se este é realmente o ambiente.csv."
                )

            registro = next(leitor, None)

            if registro and registro.get("ambiente_id"):
                return registro["ambiente_id"]

    garantir_cabecalho(caminho, CAMPOS_AMBIENTE)

    with caminho.open(
        "a", newline="", encoding="utf-8"
    ) as arquivo:
        writer = csv.DictWriter(
            arquivo, fieldnames=CAMPOS_AMBIENTE
        )
        writer.writerow(ambiente)
        arquivo.flush()
        os.fsync(arquivo.fileno())

    return ambiente["ambiente_id"]


# ---------------------------------------------------------------------
# Preparação e execução do worker
# ---------------------------------------------------------------------

def preparar_worker(tarefa, args):
    """Monta o comando de execução do worker."""
    inst = tarefa["instancia"]

    nome_json = (
        f"{inst['nome']}_{tarefa['algoritmo']}_"
        f"{tarefa['estrategia']}_{tarefa['modo']}_"
        f"{tarefa['repeticao']}.json"
    )

    saida = Path(args.resultados) / nome_json
    saida.parent.mkdir(parents=True, exist_ok=True)

    comando = [
        sys.executable,
        str(RAIZ / "worker.py"),
        "--instancia", inst["nome"],
        "--algoritmo", tarefa["algoritmo"],
        "--estrategia", tarefa["estrategia"],
        "--modo", tarefa["modo"],
        "--rep", str(tarefa["repeticao"]),
        "--limite_s", str(args.limite_s),
        "--saida", str(saida),
        "--edge-weight-type", inst["edge_weight_type"],
        "--diretorio-instancias", str(args.diretorio_instancias),
    ]

    return comando, saida


def linha_na(tarefa, ambiente_id, motivo):
    """Cria uma linha padronizada para uma execução sem resultado."""
    inst = tarefa["instancia"]
    linha = {campo: NA for campo in CAMPOS_EXECUCOES}

    linha.update({
        "instancia": inst["nome"],
        "N": inst["N"],
        "grupo_N": inst["grupo_N"],
        "edge_weight_type": inst["edge_weight_type"],
        "tem_coordenadas": inst["tem_coordenadas"],
        "otimo": inst["otimo"],
        "otimo_fonte": inst["otimo_fonte"],
        "algoritmo": tarefa["algoritmo"],
        "estrategia_warmstart": tarefa["estrategia"],
        "modo": tarefa["modo"],
        "repeticao": tarefa["repeticao"],
        "status": "NA",
        "motivo_NA": motivo,
        "ambiente_id": ambiente_id,
    })

    return linha


def montar_linha(tarefa, resultado, ambiente_id):
    """Transforma o resultado JSON em uma linha de execucoes.csv."""

    inst = tarefa["instancia"]
    linha = linha_na(tarefa, ambiente_id, "-")

    status = resultado.get("status", "NA")
    motivo = resultado.get("motivo_NA")

    # Execuções bem-sucedidas não possuem motivo de NA.
    if status == "OK" or not motivo:
        motivo = "-"

    linha.update({
        "custo": resultado.get("custo", NA),
        "tempo_total_s": resultado.get("tempo_total_s", NA),
        "tempo_heuristica_s": resultado.get("tempo_heuristica_s", NA),
        "nos_expandidos": resultado.get("nos_expandidos", NA),
        "nos_podados": resultado.get("nos_podados", NA),
        "profundidade_max": resultado.get("profundidade_max", NA),
        "lower_bound_final": resultado.get("lower_bound_final", NA),
        "melhor_ub_final": resultado.get("melhor_ub_final", NA),
        "pico_memoria_MB": resultado.get("pico_memoria_MB", NA),
        "status": status,
        "motivo_NA": motivo,
        "violacoes_reais": resultado.get("violacoes_reais", NA),
        "violacoes_arredondadas": resultado.get(
            "violacoes_arredondadas", NA
        ),
    })

    violacoes = linha["violacoes_arredondadas"]

    linha["desig_triangular"] = (
        "sim" if violacoes == 0
        else "nao" if violacoes != NA
        else NA
    )

    if linha["status"] == "OK":
        otimo = inst["otimo"]
        custo = linha["custo"]

        if otimo not in (None, NA, 0) and custo not in (None, NA):
            linha["gap"] = (
                (float(custo) - float(otimo)) / float(otimo)
            )

    return linha


def gravar_profiling(tarefa, tempos, caminho):
    """Grava profiling dos formatos de lista (BnB) e dicionário (TAT/CHR)."""
    if isinstance(tempos, dict):
        registros = [
            {
                "subrotina": nome,
                "chamadas": NA,
                "tempo_total_s": tempo,
                "memoria_pico_MB": NA,
            }
            for nome, tempo in tempos.items()
        ]

    elif isinstance(tempos, (list, tuple)):
        registros = tempos

    else:
        raise TypeError(
            f"Formato de profiling não reconhecido: {type(tempos)}"
        )

    for dados in registros:
        linha = {
            "instancia": tarefa["instancia"]["nome"],
            "algoritmo": tarefa["algoritmo"],
            "estrategia": tarefa["estrategia"],
            "modo": tarefa["modo"],
            "repeticao": tarefa["repeticao"],
            "subrotina": dados["subrotina"],
            "chamadas": dados.get("chamadas", NA),
            "tempo_s": dados.get("tempo_total_s", NA),
            "memoria_pico_MB": dados.get("memoria_pico_MB", NA),
        }

        append_csv(caminho, CAMPOS_PROFILING, linha)


def salvar_stderr(tarefa, stderr, diretorio_logs):
    """Salva o stderr de uma execução com falha."""
    diretorio_logs = Path(diretorio_logs)
    diretorio_logs.mkdir(parents=True, exist_ok=True)

    nome = (
        f"{tarefa['instancia']['nome']}_"
        f"{tarefa['algoritmo']}_"
        f"{tarefa['estrategia']}_"
        f"{tarefa['modo']}_"
        f"{tarefa['repeticao']}.err"
    )

    with (diretorio_logs / nome).open("w", encoding="utf-8") as arquivo:
        arquivo.write(stderr or "Subprocesso terminou com erro sem stderr.")


def executar_tarefa(tarefa, args, ambiente_id):
    """Executa o worker e interpreta seu JSON de saída."""
    comando, caminho_json = preparar_worker(tarefa, args)
    inicio = time.perf_counter()

    try:
        processo = subprocess.run(
            comando,
            capture_output=True,
            text=True,
            timeout=args.timeout_externo,
            check=False,
        )

        if processo.returncode != 0:
            salvar_stderr(tarefa, processo.stderr, args.logs)

            return {
                "linha": linha_na(tarefa, ambiente_id, "erro"),
                "tempos": {},
                "duracao": time.perf_counter() - inicio,
            }

        if not caminho_json.exists():
            salvar_stderr(
                tarefa,
                processo.stderr,
                args.logs,
            )

            return {
                "linha": linha_na(tarefa, ambiente_id, "erro"),
                "tempos": {},
                "duracao": time.perf_counter() - inicio,
            }

        with caminho_json.open(encoding="utf-8") as arquivo:
            dados = json.load(arquivo)

        resultado = dados["resultado"]

        return {
            "linha": montar_linha(tarefa, resultado, ambiente_id),
            "tempos": dados.get("tempos", {}),
            "duracao": time.perf_counter() - inicio,
        }

    except subprocess.TimeoutExpired as erro:
        stderr = erro.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

        salvar_stderr(tarefa, stderr, args.logs)

        return {
            "linha": linha_na(tarefa, ambiente_id, "timeout_externo"),
            "tempos": {},
            "duracao": time.perf_counter() - inicio,
        }


# ---------------------------------------------------------------------
# Registro e apresentação
# ---------------------------------------------------------------------

def registrar(tarefa, retorno, args, ambiente_id, piloto=False):
    """Registra o resultado e seu profiling, respeitando o modo piloto."""
    linha = retorno["linha"]

    if not piloto:
        append_csv(args.execucoes, CAMPOS_EXECUCOES, linha)

    if (
        linha["motivo_NA"] not in (
            "timeout_externo", "erro", "nao_aplicavel"
        )
        and retorno["tempos"]
    ):
        gravar_profiling(
            tarefa,
            retorno["tempos"],
            args.profiling,
        )

    gap = linha["gap"]

    print(
        f"[{tarefa['instancia']['nome']}] "
        f"{tarefa['algoritmo']}/{tarefa['estrategia']} "
        f"rep={tarefa['repeticao']} "
        f"status={linha['status']} "
        f"custo={linha['custo']} "
        f"gap={gap} "
        f"tempo={linha['tempo_total_s']}",
        flush=True,
    )


def imprimir_resumo_piloto(resultados):
    """Apresenta o resumo final das sete tarefas do piloto."""
    print("\nResumo do piloto:")
    print(
        f"{'Algoritmo':<10}"
        f"{'Estratégia':<14}"
        f"{'Status':<10}"
        f"{'Custo':<12}"
        f"{'Gap':<14}"
        f"{'Tempo (s)':>14}"
    )

    for tarefa, retorno in resultados:
        linha = retorno["linha"]

        print(
            f"{tarefa['algoritmo']:<10}"
            f"{tarefa['estrategia']:<14}"
            f"{linha['status']:<10}"
            f"{str(linha['custo']):<12}"
            f"{str(linha['gap']):<14}"
            f"{str(linha['tempo_total_s']):>14}"
        )


# ---------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Harness experimental do TSP."
    )

    parser.add_argument("--config", default="experimentos.json")
    parser.add_argument(
        "--modo", choices=["tempo", "memoria"], default="tempo"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--piloto", default=None)
    parser.add_argument("--instancia", default=None)

    parser.add_argument("--limite_s", type=float, default=1800)
    parser.add_argument("--timeout_externo", type=float, default=1900)

    parser.add_argument(
        "--diretorio-instancias",
        type=Path,
        default=RAIZ / "instancias" / "_extraidos",
    )

    parser.add_argument("--resultados", default="resultados")
    parser.add_argument("--logs", default="logs")

    parser.add_argument("--execucoes", default="resultados/execucoes.csv")
    parser.add_argument("--profiling", default="resultados/profiling.csv")
    parser.add_argument("--ambiente", default="resultados/ambiente.csv")

    parser.add_argument(
        "--permitir-paralelismo-tempo",
        action="store_true",
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers deve ser pelo menos 1")

    if args.reps < 1:
        parser.error("--reps deve ser pelo menos 1")

    if (
        args.modo == "tempo"
        and args.workers > 1
        and not args.permitir_paralelismo_tempo
    ):
        parser.error(
            "Para paralelismo no modo tempo, informe "
            "--permitir-paralelismo-tempo."
        )

    config = ler_config(args.config)
    instancias = sorted(
        config["instancias"],
        key=lambda inst: int(inst["N"]),
    )

    piloto = args.piloto is not None
    reps = 1 if piloto else args.reps

    nome_instancia = args.piloto or args.instancia

    if nome_instancia is not None:
        instancias = [
            inst for inst in instancias
            if inst["nome"] == nome_instancia
        ]

        if not instancias:
            parser.error(f"Instância não encontrada: {nome_instancia}")


    if piloto:
        instancias = [
            inst for inst in instancias
            if inst["nome"] == args.piloto
        ]

        if not instancias:
            parser.error(f"Instância não encontrada: {args.piloto}")

        args.resultados = str(Path(args.resultados) / "piloto")

        # O piloto não reutiliza resultados de execuções anteriores.
        Path(args.resultados).mkdir(parents=True, exist_ok=True)

        # Reinicia somente o profiling específico do piloto.
        args.profiling = str(Path(args.resultados) / "resultados" / "profiling.csv")
        garantir_cabecalho(args.profiling, CAMPOS_PROFILING)

        with Path(args.profiling).open(
            "w", newline="", encoding="utf-8"
        ) as arquivo:
            csv.DictWriter(
                arquivo, fieldnames=CAMPOS_PROFILING
            ).writeheader()

    tarefas_tuplas = gerar_tarefas(instancias, reps, args.modo)
    validar_quantidade_tarefas(tarefas_tuplas, reps)

    tarefas = [
        tarefa_para_dict(tarefa)
        for tarefa in tarefas_tuplas
    ]

    # O piloto não consulta CSV de retomada nem grava execucoes.csv.
    if piloto:
        concluidas = set()
        ambiente_id = str(uuid.uuid4())

    else:
        ambiente = coletar_ambiente(args.modo, args.workers)
        ambiente_id = obter_ou_gravar_ambiente(
            args.ambiente, ambiente
        )

        garantir_cabecalho(args.execucoes, CAMPOS_EXECUCOES)
        garantir_cabecalho(args.profiling, CAMPOS_PROFILING)

        concluidas = carregar_chaves(args.execucoes)

    imprimir_resumo(
        tarefas_tuplas,
        concluidas,
        args.modo,
        reps,
    )

    pendentes = [
        tarefa for tarefa in tarefas
        if chave_tarefa(tarefa) not in concluidas
    ]

    if not pendentes:
        print("\nNenhuma tarefa pendente.")
        return

    inicio_geral = time.perf_counter()
    resultados_piloto = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futuros = {
            pool.submit(
                executar_tarefa,
                tarefa,
                args,
                ambiente_id,
            ): tarefa
            for tarefa in pendentes
        }

        for futuro in as_completed(futuros):
            tarefa = futuros[futuro]

            try:
                retorno = futuro.result()

            except Exception as erro:
                print(f"Erro no harness: {erro}", flush=True)

                retorno = {
                    "linha": linha_na(tarefa, ambiente_id, "erro"),
                    "tempos": {},
                    "duracao": 0,
                }

                salvar_stderr(
                    tarefa,
                    repr(erro),
                    args.logs,
                )

            registrar(
                tarefa,
                retorno,
                args,
                ambiente_id,
                piloto=piloto,
            )

            if piloto:
                resultados_piloto.append((tarefa, retorno))

    duracao_total = time.perf_counter() - inicio_geral

    if piloto:
        imprimir_resumo_piloto(resultados_piloto)

    print(
        f"\nExperimento finalizado em {duracao_total:.2f} segundos."
    )


if __name__ == "__main__":
    main()
