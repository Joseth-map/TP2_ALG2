
import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from aproximativos import (
    ler_tsplib,
    construir_matriz,
    twice_around_the_tree,
    christofides,
)

from branch_bound import (
    branch_and_bound,
    verificar_triangular_real,
)

NA = "NA"


def converter_json(obj):
    """Converte tipos NumPy e valores não serializáveis para JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Tipo não serializável: {type(obj)}")


def resultado_vazio(motivo):
    """Cria um resultado padronizado para execuções não realizadas."""
    return {
        "custo": NA,
        "tour": NA,
        "status": "NA",
        "motivo_NA": motivo,
        "tempo_total_s": NA,
        "tempo_heuristica_s": NA,
        "nos_expandidos": NA,
        "nos_podados": NA,
        "profundidade_max": NA,
        "lower_bound_final": NA,
        "melhor_ub_final": NA,
        "pico_memoria_MB": NA,
        "violacoes_reais": NA,
        "violacoes_arredondadas": NA,
    }


def executar_aproximativo(
    D,
    algoritmo,
    tempos=None,
    medir_memoria=False
):
    """
    Adaptador para os algoritmos aproximativos.
    """

    if algoritmo == "TAT":
        funcao = twice_around_the_tree
    elif algoritmo == "CHR":
        funcao = christofides
    else:
        raise ValueError(f"Algoritmo desconhecido: {algoritmo}")

    tour, custo = funcao(
        D,
        tempos=tempos,
        medir_memoria=medir_memoria
    )

    return {
        "custo": custo,
        "tour": tour,
        "status": "OK",
        "motivo_NA": "-",
        "tempo_heuristica_s": NA,
        "nos_expandidos": NA,
        "nos_podados": NA,
        "profundidade_max": NA,
        "lower_bound_final": NA,
        "melhor_ub_final": NA,
        "pico_memoria_MB": NA,
        "violacoes_reais": NA,
        "violacoes_arredondadas": NA,
    }


def executar(args):
    """Executa uma tarefa e retorna resultado e profiling."""

    # O caso EXPLICIT + fecho é não aplicável.
    if (
        args.algoritmo == "BnB"
        and args.estrategia == "fecho"
        and args.edge_weight_type.upper() == "EXPLICIT"
    ):
        return resultado_vazio("nao_aplicavel"), {}

    caminho = Path(args.diretorio_instancias) / f"{args.instancia}.tsp"

    inst = ler_tsplib(str(caminho))
    D = construir_matriz(inst)

    if args.modo == "memoria":
        tracemalloc.start()

    inicio = time.perf_counter()

    try:
        if args.algoritmo == "BnB":
            resultado = branch_and_bound(
                inst,
                D,
                estrategia=args.estrategia,
                limite_s=args.limite_s,
            )
            # O BnB já fornece o profiling interno.
            tempos = resultado.pop("tempos", [])        

        else:
            tempos = {}

            resultado = executar_aproximativo(
                D,
                args.algoritmo,
                tempos=tempos,
                medir_memoria=(args.modo == "memoria")
            )

    finally:
        tempo_total = time.perf_counter() - inicio

        if args.modo == "memoria":
            _, pico = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # O pico é medido exclusivamente dentro do algoritmo.
            resultado["pico_memoria_MB"] = pico / (1024 ** 2)

    resultado["tempo_total_s"] = tempo_total

    # Garantia de que todos os campos esperados existem.
    campos = resultado_vazio("-")
    campos.update(resultado)
    resultado = campos

    return resultado, tempos


def main():
    parser = argparse.ArgumentParser(
        description="Worker de execução experimental do TSP."
    )

    parser.add_argument("--instancia", required=True)
    parser.add_argument(
        "--algoritmo", choices=["BnB", "TAT", "CHR"], required=True
    )
    parser.add_argument("--estrategia", default="NA")
    parser.add_argument(
        "--modo", choices=["tempo", "memoria"], required=True
    )
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--limite_s", type=float, default=1800)
    parser.add_argument("--saida", required=True)
    parser.add_argument("--edge-weight-type", default="EUC_2D")
    RAIZ = Path(__file__).resolve().parent

    parser.add_argument(
        "--diretorio-instancias",
        type=Path,
        default=RAIZ / "instancias" / "_extraidos",
    )

    args = parser.parse_args()

    resultado, tempos = executar(args)

    saida = {
        "instancia": args.instancia,
        "algoritmo": args.algoritmo,
        "estrategia": args.estrategia,
        "modo": args.modo,
        "repeticao": args.rep,
        "resultado": resultado,
        "tempos": tempos,
    }

    caminho_saida = Path(args.saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    with caminho_saida.open("w", encoding="utf-8") as arquivo:
        json.dump(
            saida,
            arquivo,
            indent=2,
            ensure_ascii=False,
            default=converter_json,
        )

    print(json.dumps({
        "status": resultado["status"],
        "custo": resultado["custo"],
        "tempo_total_s": resultado["tempo_total_s"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
