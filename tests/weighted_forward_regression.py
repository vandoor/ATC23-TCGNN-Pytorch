#!/usr/bin/env python3

import importlib

import torch


BLOCK_HEIGHT = 16
BLOCK_WIDTH = 8
ATOL = 1.0e-2
RTOL = 1.0e-2


def load_tcgnn():
    module = importlib.import_module("TCGNN")
    required = ("preprocess", "forward", "forward_with_value")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AssertionError("missing TCGNN symbols: {}".format(missing))
    print("TCGNN_MODULE={}".format(module.__file__))
    print("TCGNN_SYMBOLS={}".format(",".join(required)))
    return module


def make_case(tcgnn):
    row_pointer = torch.tensor(
        [0, 2, 3] + [3] * 12 + [4, 5],
        dtype=torch.int32,
    )
    column_index = torch.tensor([1, 3, 0, 2, 15], dtype=torch.int32)
    values = torch.tensor([0.25, -1.5, 2.0, 3.25, -0.75], dtype=torch.float32)
    num_nodes = row_pointer.numel() - 1
    num_edges = column_index.numel()
    block_partition = torch.zeros(1, dtype=torch.int32)
    edge_to_column = torch.zeros(num_edges, dtype=torch.int32)
    edge_to_row = torch.zeros(num_edges, dtype=torch.int32)
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
    return tuple(
        tensor.cuda().contiguous()
        for tensor in (
            row_pointer,
            column_index,
            values,
            block_partition,
            edge_to_column,
            edge_to_row,
        )
    )


def dense_reference(row_pointer, column_index, values, input):
    num_nodes = row_pointer.numel() - 1
    dense = torch.zeros(
        (num_nodes, num_nodes), dtype=torch.float32, device=input.device
    )
    row_pointer_cpu = row_pointer.cpu().tolist()
    column_index_cpu = column_index.cpu().tolist()
    values_cpu = values.cpu().tolist()
    for row in range(num_nodes):
        for edge_index in range(row_pointer_cpu[row], row_pointer_cpu[row + 1]):
            dense[row, column_index_cpu[edge_index]] = values_cpu[edge_index]
    return torch.matmul(dense, input)


def assert_close(actual, expected, name):
    if not torch.allclose(actual, expected, atol=ATOL, rtol=RTOL):
        max_abs_err = (actual - expected).abs().max().item()
        raise AssertionError("{} max_abs_err={}".format(name, max_abs_err))


def expect_runtime_error(call, message_fragment, name):
    try:
        call()
    except RuntimeError as exc:
        if message_fragment not in str(exc):
            raise AssertionError(
                "{} raised unexpected error: {}".format(name, exc)
            )
        return
    raise AssertionError("{} did not raise RuntimeError".format(name))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tcgnn = load_tcgnn()
    (
        row_pointer,
        column_index,
        values,
        block_partition,
        edge_to_column,
        edge_to_row,
    ) = make_case(tcgnn)
    input = (
        torch.arange(BLOCK_HEIGHT * BLOCK_HEIGHT, device="cuda", dtype=torch.float32)
        .reshape(BLOCK_HEIGHT, BLOCK_HEIGHT)
        .div_(32.0)
        .sub_(2.0)
        .contiguous()
    )

    weighted = tcgnn.forward_with_value(
        input,
        values,
        row_pointer,
        column_index,
        block_partition,
        edge_to_column,
        edge_to_row,
    )[0]
    expected = dense_reference(row_pointer, column_index, values, input)
    assert_close(weighted, expected, "weighted forward")

    unweighted = tcgnn.forward(
        input,
        row_pointer,
        column_index,
        block_partition,
        edge_to_column,
        edge_to_row,
    )[0]
    unit_weighted = tcgnn.forward_with_value(
        input,
        torch.ones_like(values),
        row_pointer,
        column_index,
        block_partition,
        edge_to_column,
        edge_to_row,
    )[0]
    assert_close(unit_weighted, unweighted, "unit weighted vs forward")
    torch.cuda.synchronize()
    print(
        "WEIGHTED_FORWARD_OK nnz={} max_abs_err={:.8f}".format(
            values.numel(), (weighted - expected).abs().max().item()
        )
    )

    def call_with(test_values):
        return tcgnn.forward_with_value(
            input,
            test_values,
            row_pointer,
            column_index,
            block_partition,
            edge_to_column,
            edge_to_row,
        )

    expect_runtime_error(
        lambda: call_with(values.cpu()), "values must be a CUDA tensor", "values device"
    )
    expect_runtime_error(
        lambda: call_with(torch.arange(10, device="cuda", dtype=torch.float32)[::2]),
        "values must be contiguous",
        "values contiguous",
    )
    expect_runtime_error(
        lambda: call_with(values.to(torch.float64)),
        "values must have dtype float32",
        "values dtype",
    )
    expect_runtime_error(
        lambda: call_with(values.reshape(1, -1)),
        "values must be one-dimensional",
        "values rank",
    )
    expect_runtime_error(
        lambda: call_with(values[:-1].contiguous()),
        "values length must equal CSR nnz",
        "values length",
    )
    print("VALUES_VALIDATION_OK cases=5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
