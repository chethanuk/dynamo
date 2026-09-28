# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pins the Go -> Python type mapping of generate_pydantic_from_go.py.

The generator is stdlib-only and lives outside any importable package, so it is
loaded by path.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from generate_pydantic_from_go import GoToPydanticConverter  # noqa: E402


@pytest.mark.unit
@pytest.mark.parametrize(
    ("go_type", "is_pointer", "is_optional", "expected"),
    [
        ("string", False, False, "str"),
        ("int64", False, False, "int"),
        ("bool", True, False, "bool | None"),
        ("[]string", False, False, "list[str]"),
        ("[]string", False, True, "list[str] | None"),
        # Pointer-to-slice: the slice branch ignores is_pointer. Pinned so the
        # asymmetry that test_pointer_slice_without_omitempty_is_nullable
        # exercises at emission level stays visible here.
        ("[]string", True, False, "list[str]"),
        ("map[string]string", False, False, "dict[str, str]"),
        ("map[string]string", False, True, "dict[str, str] | None"),
        ("runtime.RawExtension", False, False, "dict[str, Any]"),
        ("runtime.RawExtension", False, True, "dict[str, Any] | None"),
        ("apiextensionsv1.JSON", False, False, "Any"),
        ("apiextensionsv1.JSON", False, True, "Any | None"),
        ("[]map[string][]int64", False, False, "list[dict[str, list[int]]]"),
        ("UnknownGoType", False, False, "UnknownGoType"),
        ("UnknownGoType", True, False, "UnknownGoType | None"),
    ],
)
def test_go_type_to_python(go_type, is_pointer, is_optional, expected):
    converter = GoToPydanticConverter()
    assert converter._go_type_to_python(go_type, is_pointer, is_optional) == expected


@pytest.mark.unit
def test_pointer_slice_without_omitempty_is_nullable(tmp_path):
    """A Go pointer field can be nil, so it must emit an optional Python type.

    Emission-level, because the emitter computes nullability itself rather than
    delegating it to _go_type_to_python.
    """
    go_file = tmp_path / "mocker_types.go"
    go_file.write_text(
        "package v1beta1\n"
        "\n"
        "type MockerSpec struct {\n"
        '\tTags *[]string `json:"tags"`\n'
        "}\n"
    )

    converter = GoToPydanticConverter()
    converter.parse_go_file(go_file)
    emitted = converter.generate_pydantic()

    assert "    tags: list[str] | None" in emitted
