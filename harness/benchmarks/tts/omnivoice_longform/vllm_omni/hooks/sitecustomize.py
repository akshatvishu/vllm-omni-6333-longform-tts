# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install diagnostics after the normal pipeline import in each worker."""

import importlib.abc
import importlib.machinery
import os
import sys

_TARGET = "vllm_omni.diffusion.models.omnivoice.pipeline_omnivoice"


class DiagnosticLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        # Delay imports until platform initialization and pipeline loading finish.
        from benchmarks.tts.omnivoice_longform.vllm_omni.diagnostics import install

        install()


class DiagnosticFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = DiagnosticLoader(spec.loader)
        return spec


if os.environ.get("OMNIVOICE_DIAGNOSTICS_DIR"):
    sys.meta_path.insert(0, DiagnosticFinder())
