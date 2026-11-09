"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    matrices = np.asarray(matrices, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    return np.einsum("pij,pj->i", matrices, vectors[:, :, 0])[:, None]


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    matrix = np.asarray(matrix)
    return (matrix > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    matrix = np.asarray(matrix)
    return [list(dict.fromkeys(row.tolist())) for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    matrix = np.asarray(matrix)
    return [list(dict.fromkeys(matrix[:, j].tolist())) for j in range(matrix.shape[1])]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    row_variances = matrix.var(axis=1)
    column_variances = matrix.var(axis=0)
    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        column_means=column_means,
        row_variances=row_variances,
        column_variances=column_variances,
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    grid = np.indices((rows, columns)).sum(axis=0) % 2
    return np.where(grid == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    x0 = (image_width - width) // 2
    y0 = (image_height - height) // 2
    image[y0:y0 + height, x0:x0 + width] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    x0, y0 = image_width / 2, image_height / 2
    x = np.arange(image_width)
    y = np.arange(image_height)
    X, Y = np.meshgrid(x, y)
    mask = (X - x0) ** 2 / semi_axis_x ** 2 + (Y - y0) ** 2 / semi_axis_y ** 2 <= 1
    image[mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    variance = float(values.var())
    std = float(values.std())
    if n < 3:
        local_maxima_indices = []
        local_minima_indices = []
    else:
        local_maxima_indices = np.where((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:]))[0] + 1
        local_minima_indices = np.where((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:]))[0] + 1
    if window > n:
        moving_average = np.array([])
    else:
        moving_average = np.convolve(values, np.ones(window) / window, mode="valid")
    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima_indices=local_maxima_indices,
        local_minima_indices=local_minima_indices,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int)
    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0
    result = np.zeros((len(labels), class_count), dtype=int)
    result[np.arange(len(labels)), labels] = 1
    return result
