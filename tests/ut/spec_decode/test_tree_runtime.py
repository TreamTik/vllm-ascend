from vllm_ascend.spec_decode.tree_runtime import build_spec_tree_runtime


def test_build_spec_tree_runtime_with_branching_tree():
    runtime = build_spec_tree_runtime(((0,), (1,), (0, 0), (0, 1), (1, 0)))

    assert runtime.node_count == 5
    assert runtime.parent_index == (-1, -1, 0, 0, 1)
    assert runtime.depth == (1, 1, 2, 2, 2)
    assert runtime.children == ((2, 3), (4,), (), (), ())
    assert runtime.nodes_by_depth == ((0, 1), (2, 3, 4))
    assert runtime.leaf_indices == (2, 3, 4)


def test_build_spec_tree_runtime_with_uneven_depth():
    runtime = build_spec_tree_runtime(((0,), (1,), (1, 0), (1, 1), (1, 1, 0)))

    assert runtime.parent_index == (-1, -1, 1, 1, 3)
    assert runtime.depth == (1, 1, 2, 2, 3)
    assert runtime.nodes_by_depth == ((0, 1), (2, 3), (4,))
    assert runtime.leaf_indices == (0, 2, 4)


def test_build_spec_tree_runtime_rejects_non_parent_closed_layout():
    try:
        build_spec_tree_runtime(((0, 0),))
    except ValueError as exc:
        assert "parent path" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing parent path")
