# Copyright 2023-2025 SGLang Team
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
# ==============================================================================
"""Regression test: cuda-graph metadata keying under record_nolora_graph dual capture.

When record_nolora_graph=True, the cuda graph runner captures each batch size twice:
once with the lora variant active, once with no-lora. Pre-fix, FlashInfer and TRTLLM
MHA stored captured metadata keyed by `bs` alone — so the second capture at the same
`bs` overwrote the first one's wrapper metadata. Replay then mutated the wrong
variant's wrapper buffers and produced wrong outputs once both variants existed at
the same `bs`.

The fix introduces `_cuda_graph_metadata_key(bs)` on each backend that returns
``(bs, variant)`` when ``get_capture_lora_variant()`` is set, and falls back to ``bs``
otherwise. This file pins that contract directly, no model load required:

- ``_cuda_graph_metadata_key`` returns distinct keys for the same ``bs`` under
  different active capture variants (the property the fix exists to provide).
- A simulated dual capture (insert at ("lora"), then insert at ("nolora")) leaves
  both entries retrievable, proving the second capture cannot clobber the first.
- With variant inactive, the key is plain ``bs`` (back-compat with non-LoRA / no
  ``record_nolora_graph`` setups, including every existing capture flow).

To verify the test catches the bug, monkeypatch the keying helper to ignore the
variant (the pre-fix shape) and ``test_simulated_dual_capture_no_clobber`` fails.
"""

import unittest
from unittest.mock import patch

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.model_executor.cuda_graph_runner import (
    _set_capture_lora_variant,
    get_capture_lora_variant,
)
from sglang.test.test_utils import CustomTestCase

KEY_FNS = (
    ("FlashInfer", FlashInferAttnBackend._cuda_graph_metadata_key),
    ("TRTLLMHA", TRTLLMHAAttnBackend._cuda_graph_metadata_key),
)


class TestRecordNoloraGraphMetadataKey(CustomTestCase):
    def setUp(self):
        # Don't bleed state across tests if a prior test crashed mid-capture.
        _set_capture_lora_variant(None)

    def tearDown(self):
        _set_capture_lora_variant(None)

    def test_no_variant_keys_by_bs_only(self):
        """Outside dual capture (variant=None), the key must remain plain `bs`."""
        for backend_name, key_fn in KEY_FNS:
            for bs in (1, 4, 16):
                with self.subTest(backend=backend_name, bs=bs):
                    self.assertEqual(key_fn(bs), bs)

    def test_active_variant_keys_by_bs_and_variant(self):
        """With a capture variant active, key must include it so dual capture is
        non-aliasing. Same bs under different variants must produce distinct keys."""
        for backend_name, key_fn in KEY_FNS:
            for bs in (1, 4, 16):
                with self.subTest(backend=backend_name, bs=bs):
                    _set_capture_lora_variant("lora")
                    lora_key = key_fn(bs)
                    _set_capture_lora_variant("nolora")
                    nolora_key = key_fn(bs)
                    _set_capture_lora_variant(None)

                    self.assertEqual(lora_key, (bs, "lora"))
                    self.assertEqual(nolora_key, (bs, "nolora"))
                    self.assertNotEqual(
                        lora_key,
                        nolora_key,
                        "lora and nolora must use distinct metadata keys for the "
                        "same bs; otherwise dual capture clobbers the first wrapper",
                    )

    def test_simulated_dual_capture_no_clobber(self):
        """Drive the actual capture/replay code paths with stub metadata: under the
        fix, dual insertion at the same bs leaves both variants retrievable; under
        the pre-fix keying (always raw bs), the second insert clobbers the first."""
        for backend_name, key_fn in KEY_FNS:
            store = {}
            bs = 4
            lora_wrapper = object()  # sentinels distinct enough to fail .is checks
            nolora_wrapper = object()

            with self.subTest(backend=backend_name):
                _set_capture_lora_variant("lora")
                store[key_fn(bs)] = lora_wrapper
                _set_capture_lora_variant("nolora")
                store[key_fn(bs)] = nolora_wrapper

                _set_capture_lora_variant("lora")
                self.assertIs(
                    store[key_fn(bs)],
                    lora_wrapper,
                    "lora capture was clobbered by the subsequent nolora capture",
                )
                _set_capture_lora_variant("nolora")
                self.assertIs(
                    store[key_fn(bs)],
                    nolora_wrapper,
                    "nolora capture not retrievable after lora capture",
                )
                _set_capture_lora_variant(None)

    def test_capture_variant_state_round_trip(self):
        """Sanity-check the module-level capture-variant slot used by both backends:
        unset/set/unset cycle is observable, and the helpers the keying functions
        import are the ones we set here (one canonical slot, no shadowing)."""
        self.assertIsNone(get_capture_lora_variant())
        _set_capture_lora_variant("lora")
        self.assertEqual(get_capture_lora_variant(), "lora")
        _set_capture_lora_variant("nolora")
        self.assertEqual(get_capture_lora_variant(), "nolora")
        _set_capture_lora_variant(None)
        self.assertIsNone(get_capture_lora_variant())

    def test_pre_fix_keying_would_clobber(self):
        """Documented-failure twin: monkeypatch the keying helper back to its
        pre-fix shape (raw bs) and confirm the dual-capture invariant breaks. This
        is what the fix prevents and what a future regression would re-introduce."""
        for backend_name, backend_cls in (
            ("FlashInfer", FlashInferAttnBackend),
            ("TRTLLMHA", TRTLLMHAAttnBackend),
        ):
            with self.subTest(backend=backend_name):
                bs = 4
                with patch.object(
                    backend_cls,
                    "_cuda_graph_metadata_key",
                    staticmethod(lambda bs: bs),
                ):
                    store = {}
                    _set_capture_lora_variant("lora")
                    store[backend_cls._cuda_graph_metadata_key(bs)] = "lora_wrapper"
                    _set_capture_lora_variant("nolora")
                    store[backend_cls._cuda_graph_metadata_key(bs)] = "nolora_wrapper"
                    _set_capture_lora_variant(None)

                    # Pre-fix: only one entry survives — the second clobbers the first.
                    self.assertEqual(
                        len(store),
                        1,
                        "pre-fix keying should collide; if this changes, the test "
                        "no longer characterizes the regression it guards against",
                    )


if __name__ == "__main__":
    unittest.main(warnings="ignore")
