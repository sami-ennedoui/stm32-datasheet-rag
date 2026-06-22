"""Unit tests for the deterministic register-to-header tool.

No network or model needed. These cover the safety-critical part: the LLM
never computes an absolute address, this module does, and it validates the
data before rendering any C.
"""
import pytest

from app.regtools import build_header, ValidationError, parse_int, normalize_c_identifier


def test_build_header_computes_absolute_addresses():
    header = build_header(
        "USART1",
        "0x40011000",
        [{"name": "USART_ISR", "offset": "0x1C"}, {"name": "USART_BRR", "offset": "0x0C"}],
    )
    assert "#ifndef USART1_REGS_H" in header
    assert "#define USART1_BASE" in header
    assert "0x40011000u" in header
    # absolute addresses are base + offset, computed here, not by the model
    assert "0x4001100Cu" in header  # BRR = 0x40011000 + 0x0C
    assert "0x4001101Cu" in header  # ISR = 0x40011000 + 0x1C
    # registers are emitted sorted by offset
    assert header.index("USART_BRR") < header.index("USART_ISR")
    assert header.rstrip().endswith("#endif /* USART1_REGS_H */")


def test_build_header_accepts_int_offsets_and_base():
    header = build_header("TIM2", 0x40000000, [{"name": "TIM_CR1", "offset": 0}])
    assert "0x40000000u" in header


def test_build_header_rejects_conflicting_duplicate():
    with pytest.raises(ValidationError):
        build_header(
            "X",
            "0x1000",
            [{"name": "FOO", "offset": "0x0"}, {"name": "FOO", "offset": "0x4"}],
        )


def test_build_header_normalizes_messy_register_name():
    header = build_header("ADC1", "0x40022000", [{"name": "ADC ISR (status)", "offset": "0x00"}])
    assert "ADC_ISR_STATUS" in header


def test_build_header_rejects_empty_register_list():
    with pytest.raises(ValidationError):
        build_header("EMPTY", "0x1000", [])


def test_parse_int_handles_hex_and_decimal():
    assert parse_int("0x40011000") == 0x40011000
    assert parse_int("0x0C") == 12
    assert parse_int("1Ch") == 0x1C
    assert parse_int(16) == 16
    assert parse_int("16") == 16


def test_normalize_c_identifier_prefixes_leading_digit():
    assert normalize_c_identifier("2COOL") == "_2COOL"
