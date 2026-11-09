import numpy as np
from grader_contracts.python_basics import *
from grader_contracts.numpy_tasks import *
import python_basics as pb
import numpy_tasks as nt

# python basics
assert pb.count_vowels(TextInput("Hello World")) == 3
assert pb.count_vowels(TextInput("")) == 0
assert pb.count_vowels(TextInput("xyz")) == 0
assert pb.has_unique_characters(TextInput("abc")) is True
assert pb.has_unique_characters(TextInput("aba")) is False
assert pb.count_one_bits(PositiveIntegerInput(5)) == 2
assert pb.count_one_bits(PositiveIntegerInput(255)) == 8
assert pb.multiplicative_persistence(PositiveIntegerInput(39)) == 3
assert pb.multiplicative_persistence(PositiveIntegerInput(4)) == 0
assert pb.multiplicative_persistence(PositiveIntegerInput(999)) == 4
assert pb.mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0
assert pb.mse(VectorPairInput([1, 2, 3], [2, 3, 4])) == 1.0
assert pb.prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"
assert pb.prime_factorization(PositiveIntegerInput(7)) == "(7)"
assert pb.pyramid(PositiveIntegerInput(14)) == 3  # 1+4+9
assert pb.pyramid(PositiveIntegerInput(30)) == 4  # 1+4+9+16
assert pb.pyramid(PositiveIntegerInput(2)) == "It is impossible"
assert pb.is_balanced_number(PositiveIntegerInput(1234006)) is True
assert pb.is_balanced_number(PositiveIntegerInput(123456)) is False
assert pb.is_balanced_number(PositiveIntegerInput(5)) is True

# numpy
mats = np.random.default_rng(0).normal(size=(3, 4, 4))
vecs = np.random.default_rng(1).normal(size=(3, 4, 1))
r = nt.sum_prod(MatrixVectorBatchInput(mats, vecs))
expected = sum(mats[i] @ vecs[i] for i in range(3))
assert r.shape == (4, 1) and np.allclose(r, expected)

M = np.array([[0.1, 0.6], [0.5, 0.9]])
b = nt.binarize(BinarizeInput(M, 0.5))
assert M[0, 0] == 0.1  # не изменена
assert np.array_equal(b, [[0, 1], [0, 1]])

mm = np.array([[1, 2, 1], [3, 3, 4]])
assert nt.unique_rows(MatrixInput(mm)) == [[1, 2], [3, 4]]
assert nt.unique_columns(MatrixInput(mm)) == [[1, 3], [2, 3], [1, 4]]

ms = nt.matrix_statistics(RandomMatrixInput(rows=10, columns=5, mean=2.0, std=0.5, seed=7))
assert ms.matrix.shape == (10, 5)
assert np.allclose(ms.row_means, ms.matrix.mean(axis=1))
assert np.allclose(ms.column_variances, ms.matrix.var(axis=0))

ch = nt.chess(ChessInput(3, 4, 0, 1))
assert np.array_equal(ch, [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])

rect = nt.draw_rectangle(RectangleInput(width=4, height=2, image_height=10, image_width=10,
                                        shape_color=(255, 0, 0), background_color=(0, 0, 0)))
assert rect.shape == (10, 10, 3)
assert tuple(rect[4, 4]) == (255, 0, 0) and tuple(rect[0, 0]) == (0, 0, 0)

ell = nt.draw_ellipse(EllipseInput(semi_axis_x=3, semi_axis_y=2, image_height=11, image_width=11,
                                   shape_color=(0, 255, 0), background_color=(0, 0, 0)))
assert ell.shape == (11, 11, 3)
assert tuple(ell[5, 5]) == (0, 255, 0) and tuple(ell[0, 0]) == (0, 0, 0)

ts = nt.analyze_time_series(TimeSeriesInput([1, 3, 2, 5, 4, 1], window=2))
assert ts.mean == np.mean([1, 3, 2, 5, 4, 1])
assert list(ts.local_maxima_indices) == [1, 3]
assert list(ts.local_minima_indices) == [2]
assert np.allclose(ts.moving_average, [2, 2.5, 3.5, 4.5, 2.5])

oh = nt.one_hot(OneHotInput([0, 2, 3, 0]))
assert np.array_equal(oh, [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
oh2 = nt.one_hot(OneHotInput([0, 2, 3, 0], class_count=6))
assert oh2.shape == (4, 6)

print("LAB1 ALL TESTS PASSED")
