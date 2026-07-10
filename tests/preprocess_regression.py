#!/usr/bin/env python3

import argparse
import importlib
import sys

import numpy as np
import torch


BLOCK_HEIGHT = 16
BLOCK_WIDTH = 8


def load_tcgnn():
    module = importlib.import_module("TCGNN")
    required = ("preprocess", "preprocess_gpu")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AssertionError("missing TCGNN symbols: {}".format(missing))
    print("TCGNN_MODULE={}".format(module.__file__))
    print("TCGNN_SYMBOLS={}".format(",".join(required)))
    return module


def run_cpu_preprocess(tcgnn, row_pointer, column_index):
    row_pointer = torch.tensor(row_pointer, dtype=torch.int32)
    column_index = torch.tensor(column_index, dtype=torch.int32)
    num_nodes = row_pointer.numel() - 1
    num_edges = column_index.numel()
    num_windows = (num_nodes + BLOCK_HEIGHT - 1) // BLOCK_HEIGHT
    block_partition = torch.full((num_windows,), -1, dtype=torch.int32)
    edge_to_column = torch.full((num_edges,), -1, dtype=torch.int32)
    edge_to_row = torch.full((num_edges,), -1, dtype=torch.int32)
    tcgnn.preprocess(
        column_index,
        row_pointer,
        num_nodes,
        BLOCK_HEIGHT,
        BLOCK_WIDTH,
        block_partition,
        edge_to_column,
        edge_to_row,
    )
    return block_partition, edge_to_column, edge_to_row


def assert_tensor(actual, expected, name):
    expected_tensor = torch.tensor(expected, dtype=torch.int32)
    if not torch.equal(actual, expected_tensor):
        raise AssertionError(
            "{} mismatch: actual={} expected={}".format(
                name, actual.tolist(), expected_tensor.tolist()
            )
        )


def test_exact_window(tcgnn):
    row_pointer = [0, 2, 3] + [3] * 13 + [4]
    block_partition, edge_to_column, edge_to_row = run_cpu_preprocess(
        tcgnn, row_pointer, [3, 1, 3, 2]
    )
    assert_tensor(block_partition, [1], "exact blockPartition")
    assert_tensor(edge_to_column, [2, 0, 2, 1], "exact edgeToColumn")
    assert_tensor(edge_to_row, [0, 0, 1, 15], "exact edgeToRow")


def test_empty_window(tcgnn):
    block_partition, edge_to_column, edge_to_row = run_cpu_preprocess(
        tcgnn, [0] * 17, []
    )
    assert_tensor(block_partition, [0], "empty blockPartition")
    assert_tensor(edge_to_column, [], "empty edgeToColumn")
    assert_tensor(edge_to_row, [], "empty edgeToRow")


def test_tail_window(tcgnn):
    row_pointer = [0, 2, 3] + [3] * 13 + [4, 7]
    block_partition, edge_to_column, edge_to_row = run_cpu_preprocess(
        tcgnn, row_pointer, [3, 1, 3, 2, 4, 4, 2]
    )
    assert_tensor(block_partition, [1, 1], "tail blockPartition")
    assert_tensor(edge_to_column, [2, 0, 2, 1, 1, 1, 0], "tail edgeToColumn")
    assert_tensor(edge_to_row, [0, 0, 1, 15, 16, 16, 16], "tail edgeToRow")


def expect_runtime_error(call, label):
    try:
        call()
    except RuntimeError:
        return
    raise AssertionError("{} did not raise RuntimeError".format(label))


def test_capacity_checks(tcgnn):
    row_pointer = torch.tensor([0] * 17, dtype=torch.int32)
    empty = torch.empty(0, dtype=torch.int32)
    one_window = torch.zeros(1, dtype=torch.int32)

    expect_runtime_error(
        lambda: tcgnn.preprocess(
            empty, row_pointer[:-1], 16, 16, 8, one_window, empty, empty
        ),
        "short nodePointer",
    )
    expect_runtime_error(
        lambda: tcgnn.preprocess(
            empty, row_pointer, 16, 16, 8, empty, empty, empty
        ),
        "short blockPartition",
    )

    nonempty_rows = torch.tensor([0, 1] + [1] * 15, dtype=torch.int32)
    one_edge = torch.tensor([0], dtype=torch.int32)
    expect_runtime_error(
        lambda: tcgnn.preprocess(
            one_edge,
            nonempty_rows,
            16,
            16,
            8,
            one_window,
            empty,
            one_edge.clone(),
        ),
        "short edgeToColumn",
    )
    expect_runtime_error(
        lambda: tcgnn.preprocess(
            one_edge,
            nonempty_rows,
            16,
            16,
            8,
            one_window,
            one_edge.clone(),
            empty,
        ),
        "short edgeToRow",
    )


def run_synthetic(tcgnn):
    test_exact_window(tcgnn)
    test_empty_window(tcgnn)
    test_tail_window(tcgnn)
    test_capacity_checks(tcgnn)
    print("SYNTHETIC_PREPROCESS_OK cases=4")


def load_igb_csr(path):
    from scipy.io import mmread

    csr = mmread(path).tocsr()
    csr.sum_duplicates()
    csr.sort_indices()
    if csr.shape[0] != csr.shape[1]:
        raise AssertionError("IGB matrix must be square, got {}".format(csr.shape))
    row_pointer = np.asarray(csr.indptr, dtype=np.int32)
    column_index = np.asarray(csr.indices, dtype=np.int32)
    return row_pointer, column_index


def run_igb(tcgnn, path, expected_nodes, gpu_only):
    row_pointer_np, column_index_np = load_igb_csr(path)
    num_nodes = row_pointer_np.size - 1
    num_edges = column_index_np.size
    num_windows = (num_nodes + BLOCK_HEIGHT - 1) // BLOCK_HEIGHT
    if num_nodes != expected_nodes:
        raise AssertionError(
            "IGB node count mismatch: actual={} expected={}".format(
                num_nodes, expected_nodes
            )
        )
    if num_windows != 6250:
        raise AssertionError("IGB window count mismatch: {}".format(num_windows))

    if not gpu_only:
        block_partition, edge_to_column, edge_to_row = run_cpu_preprocess(
            tcgnn, row_pointer_np, column_index_np
        )
        if block_partition.numel() != num_windows:
            raise AssertionError("blockPartition length mismatch")
        if edge_to_column.numel() != num_edges or edge_to_row.numel() != num_edges:
            raise AssertionError("edge metadata length mismatch")
        print(
            "IGB_CPU_OK nodes={} edges={} windows={}".format(
                num_nodes, num_edges, block_partition.numel()
            )
        )

    if torch.cuda.is_available():
        row_pointer = torch.from_numpy(row_pointer_np).cuda()
        column_index = torch.from_numpy(column_index_np).cuda()
        block_partition = torch.zeros(num_windows, dtype=torch.int32, device="cuda")
        edge_to_column = torch.zeros(num_edges, dtype=torch.int32, device="cuda")
        edge_to_row = torch.zeros(num_edges, dtype=torch.int32, device="cuda")
        tcgnn.preprocess_gpu(
            column_index,
            row_pointer,
            num_nodes,
            BLOCK_HEIGHT,
            BLOCK_WIDTH,
            block_partition,
            edge_to_column,
            edge_to_row,
        )
        torch.cuda.synchronize()
        print(
            "IGB_GPU_OK nodes={} edges={} windows={}".format(
                num_nodes, num_edges, block_partition.numel()
            )
        )
    elif gpu_only:
        raise RuntimeError("CUDA is required for --gpu-only")


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--igb-mtx")
    parser.add_argument("--expected-nodes", type=int, default=100000)
    parser.add_argument("--gpu-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    tcgnn = load_tcgnn()
    if args.igb_mtx:
        run_igb(tcgnn, args.igb_mtx, args.expected_nodes, args.gpu_only)
    else:
        run_synthetic(tcgnn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
