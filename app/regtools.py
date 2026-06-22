"""Deterministic register-to-C-header tool.

This is the safety-critical half of the agent. The language model may read a
datasheet and propose register names and offsets relative to a peripheral base,
but it never computes an absolute address. This module does the arithmetic
(base + offset), validates every name as a C identifier, rejects conflicting
duplicates, and renders the header. A hallucinated absolute address can brick
hardware, so that step stays in plain Python.

Vendored and trimmed from the ~/smolagent project so this repo has no external
dependency for the deterministic path.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


class ValidationError(ValueError):
    """Raised when register data cannot safely be rendered as C."""


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_int(value: int | str) -> int:
    """Parse a hex or decimal address, tolerant of datasheet formatting."""
    if isinstance(value, int):
        return value
    cleaned = value.strip().lower().replace("_", "").replace(" ", "")
    if not cleaned:
        raise ValidationError("empty integer value")
    if cleaned.endswith("h"):
        return int(cleaned[:-1], 16)
    if cleaned.startswith("0x"):
        return int(cleaned[2:], 16)
    if any(ch in "abcdef" for ch in cleaned):
        return int(cleaned, 16)
    return int(cleaned, 10)


def normalize_c_identifier(value: str) -> str:
    """Return a stable C macro identifier for a datasheet register name."""
    value = value.strip()
    if not value:
        raise ValidationError("empty identifier")
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", value)
    normalized = re.sub(r"_+", "_", normalized).strip("_").upper()
    if not normalized:
        raise ValidationError(f"invalid identifier: {value!r}")
    if normalized[0].isdigit():
        normalized = f"_{normalized}"
    if not _IDENTIFIER_RE.match(normalized):
        raise ValidationError(f"invalid identifier after normalization: {value!r}")
    return normalized


@dataclass(frozen=True)
class Register:
    name: str
    offset: int
    description: str | None = None

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValidationError(f"negative offset for {self.name}: {self.offset}")
        normalize_c_identifier(self.name)

    @property
    def c_name(self) -> str:
        return normalize_c_identifier(self.name)

    def absolute_address(self, base_address: int) -> int:
        return base_address + self.offset


@dataclass(frozen=True)
class Peripheral:
    name: str
    base_address: int
    registers: tuple[Register, ...]

    def __post_init__(self) -> None:
        if self.base_address < 0:
            raise ValidationError(f"negative base address for {self.name}")
        normalize_c_identifier(self.name)
        if not self.registers:
            raise ValidationError(f"{self.name} has no registers")

    @property
    def c_name(self) -> str:
        return normalize_c_identifier(self.name)

    @classmethod
    def from_registers(
        cls, name: str, base_address: int, registers: list[Register]
    ) -> "Peripheral":
        deduped: dict[str, Register] = {}
        for register in registers:
            existing = deduped.get(register.c_name)
            if existing is None:
                deduped[register.c_name] = register
                continue
            if existing.offset != register.offset:
                raise ValidationError(
                    f"register {register.c_name} appears at both "
                    f"0x{existing.offset:X} and 0x{register.offset:X}"
                )
        ordered = sorted(deduped.values(), key=lambda item: item.offset)
        return cls(name=name, base_address=base_address, registers=tuple(ordered))


@dataclass(frozen=True)
class HeaderOptions:
    guard: str | None = None
    macro_prefix: str | None = None
    hex_width: int = 8
    integer_suffix: str = "u"
    include_base: bool = True


def render_header(peripheral: Peripheral, options: HeaderOptions | None = None) -> str:
    options = options or HeaderOptions()
    guard = normalize_c_identifier(options.guard or f"{peripheral.c_name}_REGS_H")

    prefix = ""
    if options.macro_prefix:
        prefix = normalize_c_identifier(options.macro_prefix)
        if not prefix.endswith("_"):
            prefix += "_"

    macro_names: set[str] = set()
    rows: list[tuple[str, str]] = []

    if options.include_base:
        base_macro = f"{prefix}{peripheral.c_name}_BASE"
        rows.append((base_macro, _format_hex(peripheral.base_address, options)))
        macro_names.add(base_macro)

    for register in peripheral.registers:
        macro = f"{prefix}{register.c_name}"
        if macro in macro_names:
            raise ValidationError(f"duplicate macro name: {macro}")
        macro_names.add(macro)
        rows.append(
            (macro, _format_hex(register.absolute_address(peripheral.base_address), options))
        )

    column_width = max(len(name) for name, _ in rows) + 1
    lines = [f"#ifndef {guard}", f"#define {guard}", ""]
    for macro, value in rows:
        lines.append(f"#define {macro:<{column_width}} {value}")
    lines.extend(["", f"#endif /* {guard} */", ""])
    return "\n".join(lines)


def _format_hex(value: int, options: HeaderOptions) -> str:
    return f"0x{value:0{options.hex_width}X}{options.integer_suffix}"


def build_header(
    peripheral: str,
    base_address: int | str,
    registers: list[dict],
    macro_prefix: str | None = None,
) -> str:
    """High level entry point used by the agent tool.

    ``registers`` is a list of ``{"name": str, "offset": hex-or-int}``. The model
    supplies relative offsets only. Validation and address arithmetic happen here.
    """
    base = parse_int(base_address)
    regs = [
        Register(
            name=str(item["name"]),
            offset=parse_int(item["offset"]),
            description=item.get("description"),
        )
        for item in registers
    ]
    peripheral_obj = Peripheral.from_registers(
        name=peripheral, base_address=base, registers=regs
    )
    return render_header(peripheral_obj, HeaderOptions(macro_prefix=macro_prefix))
