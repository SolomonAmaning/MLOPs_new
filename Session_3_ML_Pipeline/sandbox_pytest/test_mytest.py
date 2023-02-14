from mytest import square
import pytest
def test_square_gives_correct_value():
    out=square(2)
    assert out==4
