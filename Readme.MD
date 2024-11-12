# BFGS Algorithm
## Overview

The Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm is an iterative method for solving unconstrained nonlinear optimization problems. It belongs to the family of quasi-Newton methods, which are used to find local maxima and minima of functions.

### Equations Involved

The BFGS method provides an efficient way to approximation of the inverse Hessian matrix, \($H_k$\), at each iteration. The update formula is:

$$
H_{k+1} = \left( I - \frac{s_k y_k^T}{y_k^T s_k} \right) H_k \left( I - \frac{y_k s_k^T}{y_k^T s_k} \right) + \frac{s_k s_k^T}{y_k^T s_k}
$$

where:
- \($s_k = x_{k+1} - x_k$\) is the difference in the position vector.
- \($y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$ \) is the difference in the gradient vector.

### Quasi-Newton Method

Quasi-Newton methods, like BFGS, build up an approximation to the Hessian matrix of second derivatives of the objective function, which is used to guide the search for the minimum. Unlike Newton's method, which requires the exact Hessian, quasi-Newton methods update the Hessian approximation iteratively.

### Line Search

The BFGS algorithm typically uses a line search to find an acceptable step size \(\alpha\) that satisfies certain conditions. The line search aims to ensure sufficient decrease in the objective function and to maintain the curvature condition. The search direction is derived from the hessian matrix:

### Armijo Condition

The Armijo condition ensures that the step size \(\alpha\) provides a sufficient decrease in the objective function. It is given by:

$$
f(x_k + \alpha p_k) \leq f(x_k) + c_1 \alpha \nabla f(x_k)^T p_k 
$$

where \($0 < c_1 < 1$\) is a constant (typically with a value of 0.0001), and \($p_k$\) is the search direction.

### Curvature Condition

The curvature condition ensures that the step size \(\alpha\) is not too small and that the search direction is a descent direction. It is given by:

$$
\nabla f(x_k + \alpha p_k)^T p_k \geq c_2 \nabla f(x_k)^T p_k 
$$

where \(0 < $c_2 <$ 1\) is another constant (typically at a value of 0.9).

### Further Reading


For more detailed information, refer to the following resources:
- [BFGS Algorithm on Wikipedia](https://en.wikipedia.org/wiki/BFGS_method)
- [Numerical Optimization by Jorge Nocedal and Stephen Wright](https://www.springer.com/gp/book/9780387303031)

## Implementation

For efficiency's sake, this implementation of BFGs is coded in rust, but is designed to be run in python through its use of pyo3/Maturin. This provides the utility and flexibility of python, with the enhanced computational efficiency begotten from rust.

## Quickstart

### Step 1: Install Rust

1. Open your terminal.
2. Run the following command to install Rust using `rustup` (the Rust toolchain installer):

    ```sh
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```

3. Follow the on-screen instructions to complete the installation.
4. After installation, ensure that Rust is installed correctly by running:

    ```sh
    rustc --version
    ```

    This should display the version of Rust installed.

### Step 2: Install Maturin

1. With Rust installed, you can now install Maturin. Run the following command in your terminal:

    ```sh
    pip install maturin
    ```

2. Verify the installation by checking the version of Maturin:

    ```sh
    maturin --version
    ```

    This should display the version of Maturin installed.

For more information, refer to the official documentation:
- [Rust Documentation](https://www.rust-lang.org/learn)
- [Maturin Documentation](https://maturin.rs/)

Happy coding!
