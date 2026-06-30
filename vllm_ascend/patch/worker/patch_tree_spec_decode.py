#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer


_ORIGINAL_CREATE_SPECULATIVE_DRAFTER = GPUModelRunner._create_speculative_drafter


def _patched_create_speculative_drafter(self):
    spec_config = self.speculative_config
    if (
        spec_config is not None
        and hasattr(spec_config, "use_speculative_token_tree")
        and spec_config.use_speculative_token_tree()
        and spec_config.method in ("eagle", "eagle3")
    ):
        return AscendEagleProposer(self.vllm_config, self.device, self)

    return _ORIGINAL_CREATE_SPECULATIVE_DRAFTER(self)


GPUModelRunner._create_speculative_drafter = _patched_create_speculative_drafter
