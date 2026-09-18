"""Solucao bi-objetivo do knapsack com processamento por lotes em PyTorch.

Os objetivos sao:
    - maximizar o valor total;
    - minimizar o peso total.

A enumeracao continua sendo exata. Para evitar explodir a memoria da GPU,
as combinacoes sao geradas e reduzidas em lotes.


source .venv/bin/activate
python knapsack_torch_gpu.py \
  instance/knapPI_16_20_1000_teste.csv \
  --batch-size 65536 \
  --device cuda
  
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def total_possible_solutions(n_items: int) -> int:
    """Retorna o total de subconjuntos, incluindo a mochila vazia."""
    if n_items < 0:
        raise ValueError("n_items deve ser nao negativo")
    return 1 << n_items


def _select_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pareto_frontier(points: np.ndarray) -> np.ndarray:
    """Retorna pontos nao dominados para valor maximo e peso minimo.

    Cada ponto tem o formato ``[valor_total, peso_total]``. Pontos repetidos
    sao removidos, e a ordenacao torna a filtragem linear apos o sort.
    """
    points = np.asarray(points, dtype=np.int64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    points = points.reshape(-1, 2)
    points = np.unique(points, axis=0)

    # Valor descrescente e, em empate, peso crescente.
    order = np.lexsort((points[:, 1], -points[:, 0]))
    ordered = points[order]
    weights = ordered[:, 1]

    keep = np.empty(len(ordered), dtype=bool)
    keep[0] = True
    if len(ordered) > 1:
        keep[1:] = weights[1:] < np.minimum.accumulate(weights[:-1])

    return ordered[keep]


def _as_integer_points(points: np.ndarray | list[list[int]]) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("pontos deve ter formato (N, 2): [valor, peso]")
    if not np.issubdtype(points.dtype, np.integer):
        if not np.all(np.isfinite(points)) or not np.all(points == points.astype(np.int64)):
            raise ValueError("valor e peso devem ser inteiros")
    return points.astype(np.int64, copy=False)


def no_dominated_torch(
    pontos: np.ndarray | list[list[int]],
    cap: int | None = None,
    batch_size: int = 65_536,
    device: str | torch.device | None = None,
    min_batch_size: int = 1_024,
) -> np.ndarray:
    """Calcula exatamente a frente de Pareto do knapsack.

    Parameters
    ----------
    pontos:
        Array ``(N, 2)`` com ``[valor, peso]`` de cada item.
    cap:
        Capacidade maxima. Use ``None`` para nao aplicar capacidade.
    batch_size:
        Numero inicial de combinacoes processadas por lote.
    device:
        ``"cuda"``, ``"cpu"`` ou ``None`` para deteccao automatica.
    min_batch_size:
        Limite usado ao reduzir o lote apos falta de memoria CUDA.
    """
    pontos_np = _as_integer_points(pontos)
    if batch_size < 1 or min_batch_size < 1:
        raise ValueError("batch_size e min_batch_size devem ser positivos")
    if cap is not None and cap < 0:
        raise ValueError("cap deve ser None ou um inteiro nao negativo")

    n_itens = len(pontos_np)
    if n_itens == 0:
        return np.empty((0, 2), dtype=np.int64)
    if n_itens >= 63:
        raise ValueError("enumeracao exata por bits suporta no maximo 62 itens")

    torch_device = _select_device(device)
    max_total = n_itens * int(np.max(np.abs(pontos_np)))
    if torch_device.type == "cuda" and max_total <= 2**24:
        # CUDA oferece matmul eficiente em ponto flutuante; ate 2**24,
        # float32 representa exatamente todos os inteiros.
        computation_dtype = torch.float32
    elif max_total <= np.iinfo(np.int32).max and torch_device.type != "cuda":
        computation_dtype = torch.int32
    elif max_total <= 2**53:
        # CUDA nao implementa matmul int64 em todas as versoes. Float64 ainda
        # representa exatamente inteiros ate 2**53 e e convertido apos a soma.
        computation_dtype = torch.float64
    else:
        raise ValueError("somatorios muito grandes para uma soma exata na GPU")

    pontos_tensor = torch.as_tensor(pontos_np, dtype=computation_dtype, device=torch_device)
    total_combinacoes = 1 << n_itens
    bits = torch.arange(n_itens - 1, -1, -1, dtype=torch.int64, device=torch_device)

    frontier = np.empty((0, 2), dtype=np.int64)
    current_batch_size = min(batch_size, total_combinacoes)
    inicio = 0

    while inicio < total_combinacoes:
        fim = min(inicio + current_batch_size, total_combinacoes)
        try:
            indices = torch.arange(inicio, fim, dtype=torch.int64, device=torch_device)
            mascara = ((indices[:, None] >> bits) & 1).to(computation_dtype)
            solucoes = mascara @ pontos_tensor
            if computation_dtype in (torch.float32, torch.float64):
                solucoes = solucoes.round().to(torch.int64)
            else:
                solucoes = solucoes.to(torch.int64)

            if cap is not None:
                solucoes = solucoes[solucoes[:, 1] <= cap]

            if solucoes.numel() > 0:
                solucoes_cpu = solucoes.detach().cpu().numpy()
                # Uma solucao dominada dentro de um lote nunca pode voltar a
                # dominar uma solucao de outro lote.
                frontier = pareto_frontier(np.vstack((frontier, solucoes_cpu)))

            del indices, mascara, solucoes
            inicio = fim
        except RuntimeError as error:
            is_cuda_oom = "out of memory" in str(error).lower()
            if not is_cuda_oom or torch_device.type != "cuda":
                raise
            if current_batch_size <= min_batch_size:
                raise RuntimeError(
                    "Sem memoria CUDA mesmo no lote minimo; reduza min_batch_size."
                ) from error
            current_batch_size = max(min_batch_size, current_batch_size // 2)
            torch.cuda.empty_cache()

    return frontier


def solve_instance(
    pontos: np.ndarray | list[list[int]],
    cap: int | None = None,
    batch_size: int = 65_536,
    device: str | torch.device | None = None,
) -> dict[str, int | np.ndarray]:
    """Resolve uma instancia e retorna fronteira e estatisticas da enumeracao."""
    pontos_np = _as_integer_points(pontos)
    if len(pontos_np) >= 63:
        raise ValueError("enumeracao exata por bits suporta no maximo 62 itens")

    total = total_possible_solutions(len(pontos_np))
    frontier = no_dominated_torch(
        pontos_np,
        cap=cap,
        batch_size=batch_size,
        device=device,
    )

    # A contagem de viaveis e feita separadamente para manter a API principal
    # simples e continuar contando combinacoes, nao apenas pontos distintos.
    torch_device = _select_device(device)
    points_tensor = torch.as_tensor(pontos_np, dtype=torch.int64)
    feasible = 0
    for inicio in range(0, total, batch_size):
        fim = min(inicio + batch_size, total)
        indices = torch.arange(inicio, fim, dtype=torch.int64)
        bits = torch.arange(len(pontos_np) - 1, -1, -1, dtype=torch.int64)
        masks = ((indices[:, None] >> bits) & 1).to(torch.int64)
        sums = masks @ points_tensor
        if cap is None:
            feasible += len(sums)
        else:
            feasible += int((sums[:, 1] <= cap).sum().item())

    return {
        "frontier": frontier,
        "total_solutions": total,
        "feasible_solutions": feasible,
        "items": len(pontos_np),
        "capacity": -1 if cap is None else cap,
        "device": str(torch_device),
    }


def plot_result(
    pontos: np.ndarray | list[list[int]],
    result: dict[str, int | np.ndarray],
    filename: str | Path,
) -> Path:
    """Gera um grafico de conferencia dos itens e da fronteira de Pareto."""
    pontos_np = _as_integer_points(pontos)
    frontier = np.asarray(result["frontier"])
    capacity = int(result["capacity"])
    title = (
        f"Knapsack bi-objetivo | {result['items']} itens | "
        f"solucoes: {result['total_solutions']:,} | "
        f"viaveis: {result['feasible_solutions']:,}"
    )
    if capacity >= 0:
        title += f" | capacidade: {capacity}"

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.scatter(
        pontos_np[:, 0], pontos_np[:, 1],
        color="steelblue", alpha=0.75, label="Itens [valor, peso]",
    )
    if frontier.size:
        axis.scatter(
            frontier[:, 0], frontier[:, 1],
            color="crimson", edgecolors="black", s=75,
            zorder=3, label="Fronteira nao dominada",
        )
        axis.plot(frontier[:, 0], frontier[:, 1], color="crimson", alpha=0.45)
    axis.set_title(title)
    axis.set_xlabel("Valor total / valor do item")
    axis.set_ylabel("Peso total / peso do item")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def read_instances(filename: str | Path) -> Iterator[tuple[str, np.ndarray, int | None]]:
    """Le instancias no formato CSV usado na pasta ``instance``."""
    filename = Path(filename)
    points: list[list[int]] = []
    expected_items = 0
    capacity: int | None = None
    name = filename.stem

    def emit() -> tuple[str, np.ndarray, int | None] | None:
        if not points:
            return None
        return name, np.asarray(points, dtype=np.int64), capacity

    with filename.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            if "knapPI" in line:
                name = line.replace(".csv", "").replace(",", "").strip()
                continue
            if line.startswith("-"):
                item = emit()
                if item is not None:
                    yield item
                points = []
                expected_items = 0
                capacity = None
                continue
            parts = line.replace(",", " ").split()
            if parts and parts[0].lower() == "n" and len(parts) > 1:
                expected_items = int(parts[1])
                continue
            if parts and parts[0].lower() == "c" and len(parts) > 1:
                capacity = int(parts[1])
                continue
            if len(parts) == 4:
                try:
                    points.append([int(parts[1]), int(parts[2])])
                except ValueError:
                    continue
                if expected_items and len(points) == expected_items:
                    item = emit()
                    if item is not None:
                        yield item
                    points = []
                    expected_items = 0
                    capacity = None

    item = emit()
    if item is not None:
        yield item


def solve_file(
    filename: str | Path,
    batch_size: int = 65_536,
    device: str | torch.device | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Resolve todas as instancias de um arquivo e retorna suas fronteiras."""
    results = []
    for name, pontos, capacity in read_instances(filename):
        result = solve_instance(
            pontos,
            cap=capacity,
            batch_size=batch_size,
            device=device,
        )
        results.append((name, pontos, result))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", help="arquivo CSV da instancia")
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", default="imagens")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    device = None if args.device == "auto" else args.device
    started = time.perf_counter()
    for name, pontos, result in solve_file(args.filename, args.batch_size, device):
        frontier = result["frontier"]
        print(f"{name}: {result['items']} itens")
        print(f"Total de solucoes possiveis: {result['total_solutions']:,}")
        print(f"Total de solucoes viaveis: {result['feasible_solutions']:,}")
        print(f"Pontos nao dominados: {len(frontier)}")
        print(frontier)
        if not args.no_plot:
            image = Path(args.output_dir) / f"{name}_torch_gpu.png"
            print(f"Imagem salva em: {plot_result(pontos, result, image)}")
    elapsed = time.perf_counter() - started
    print(f"Tempo total: {elapsed:.4f} segundos")


if __name__ == "__main__":
    main()