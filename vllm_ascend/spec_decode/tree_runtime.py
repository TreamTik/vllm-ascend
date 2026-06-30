#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

from dataclasses import dataclass


TreePath = tuple[object, ...]


@dataclass(frozen=True)
class SpecTreeRuntime:
    paths: tuple[TreePath, ...]
    node_count: int
    parent_index: tuple[int, ...]
    depth: tuple[int, ...]
    children: tuple[tuple[int, ...], ...]
    nodes_by_depth: tuple[tuple[int, ...], ...]
    leaf_indices: tuple[int, ...]


def build_spec_tree_runtime(paths: tuple[TreePath, ...]) -> SpecTreeRuntime:
    if not paths:
        raise ValueError("speculative_token_tree must contain at least one node.")

    if len(set(paths)) != len(paths):
        raise ValueError("speculative_token_tree contains duplicate paths.")

    path_to_index = {path: idx for idx, path in enumerate(paths)}
    parent_index: list[int] = []
    depth: list[int] = []
    children_lists: list[list[int]] = [[] for _ in paths]

    for idx, path in enumerate(paths):
        if not path:
            raise ValueError("speculative_token_tree paths must be non-empty.")

        node_depth = len(path)
        depth.append(node_depth)

        if node_depth == 1:
            parent_index.append(-1)
            continue

        parent_path = path[:-1]
        parent_idx = path_to_index.get(parent_path)
        if parent_idx is None:
            raise ValueError(
                f"speculative_token_tree path {path!r} is missing parent path {parent_path!r}."
            )

        parent_index.append(parent_idx)
        children_lists[parent_idx].append(idx)

    depth_buckets: list[list[int]] = [[] for _ in range(max(depth))]
    for idx, node_depth in enumerate(depth):
        depth_buckets[node_depth - 1].append(idx)

    return SpecTreeRuntime(
        paths=paths,
        node_count=len(paths),
        parent_index=tuple(parent_index),
        depth=tuple(depth),
        children=tuple(tuple(children) for children in children_lists),
        nodes_by_depth=tuple(tuple(nodes) for nodes in depth_buckets),
        leaf_indices=tuple(idx for idx, children in enumerate(children_lists) if not children),
    )
