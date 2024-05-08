import numpy as np

def gauss_jordan(A, B):
    n = len(B)
    for i in range(n):

        pivot = A[i][i]
        A[i] /= pivot
        B[i] /= pivot

        for j in range(n):
            if i != j:
                factor = A[j][i]
                A[j] -= factor * A[i]
                B[j] -= factor * B[i]
    return B

def seidel(A, B, initial_guess, tolerance=1e-10, max_iterations=1000):
    n = len(B)
    x = np.array(initial_guess, dtype=np.float64)
    x_new = np.zeros_like(x, dtype=np.float64)
    for _ in range(max_iterations):
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (B[i] - sum1 - sum2) / A[i, i]
        if np.allclose(x, x_new, atol=tolerance):
            return x_new
        x = x_new.copy()
    raise ValueError("Помилка.")

A_gauss = np.array([[-6.0, -8.0, -2.0, -8.0],
                           [9.0, 0.0, 8.0, 3.0],
                           [0.0, -9.0, -5.0, 9.0],
                           [-1.0, 4.0, -8.0, -4.0]], dtype=np.float64)

B_gauss = np.array([-32.0, 8.0, -2.0, -36.0], dtype=np.float64)

print("СЛАР для методу Гауса-Жордана:")
for i in range(len(B_gauss)):
    equation = " + ".join([f"{A_gauss[i][j]}x{j + 1}" for j in range(len(B_gauss))]) + f" = {B_gauss[i]}"
    print(equation)

A_seidel = np.array([[10.0, 0.0, 2.0, 4.0],
                            [2.0, 16.0, -3.0, 8.0],
                            [1.0, 5.0, 11.0, -4.0],
                            [8.0, 1.0, 6.0, -17.0]], dtype=np.float64)

B_seidel = np.array([110.0, 128.0, 102.0, 81.0], dtype=np.float64)

print("\nСЛАР для методу Зейделя:")
for i in range(len(B_seidel)):
    equation = " + ".join([f"{A_seidel[i][j]}x{j + 1}" for j in range(len(B_seidel))]) + f" = {B_seidel[i]}"
    print(equation)

solution_gauss = gauss_jordan(A_gauss, B_gauss)
print("\nРозв'язок методом Гауса-Жордана:", solution_gauss)

initial_guess = [0, 0, 0, 0]
try:
    solution_seidel = seidel(A_seidel, B_seidel, initial_guess)
    print("Розв'язок методом Зейделя:", solution_seidel)
except ValueError as e:
    print(e)
