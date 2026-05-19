"""Unit tests for core functions."""
import pytest
from toto_engine import parse_draw, extract_draw_json

def test_parse():
    sample_json = {
        "DrawNumber": 4050,
        "DrawDate": "20/05/2026",
        "WinningNumbers": [5,3,1,4,6,2],
        "AdditionalNumber": 7
    }
    row = parse_draw(sample_json)
    assert row["draw_no"] == 4050
    assert row["num1"] == 1 and row["num6"] == 6  # sorted

def test_extract():
    html = '<script>var drawResult = {"DrawNumber":4050,"DrawDate":"20/05/2026","WinningNumbers":[1,2,3,4,5,6],"AdditionalNumber":7};</script>'
    data = extract_draw_json(html)
    assert data["DrawNumber"] == 4050
