use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyModule, PyList};
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, ToPyArray};
use ndarray_linalg::Norm;

pub trait Function {
    fn value(&mut self, x: &Array1<f64>, grad: bool) -> f64;
    fn gradient(&mut self, x: &Array1<f64>) -> Array1<f64>;
    fn get_fvals(&mut self) -> Py<PyArray1<f64>>;
    fn get_xvals(&mut self) -> Py<PyArray2<f64>>;
}

#[pyclass]
pub struct Wrapper {
    callback: PyObject, 
    f_vals: Option<Vec<f64>>, 
    x_vals: Option<Array2<f64>>,
}

#[pymethods]
impl Wrapper {
    #[new]
    fn new(callback: PyObject) -> Self {
        Wrapper { callback, f_vals: None, x_vals: None } 
    }

    fn call_python_function(py_func: PyObject, py_args: &PyTuple) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let result = py_func.call1(py, py_args)?;
            let temp_var: f64 = result.extract(py)?;
            Ok(temp_var.into_py(py))
        })
    }
}

impl Wrapper {
    fn call_callback(&self, py: Python, x: &Array1<f64>) -> PyResult<f64> {
        let py_x = PyList::new(py, x.iter());
        let args = (py_x,); 
        let result = self.callback.call1(py, args)?;
        result.extract(py)
    }
}

impl Function for Wrapper {
    fn value(&mut self, x: &Array1<f64>, grad:bool) -> f64 {
        let val = Python::with_gil(|py| self.call_callback(py, x).unwrap_or(0.0));
        if grad == false {
            self.f_vals.get_or_insert_with(Vec::new).push(val.clone());
            self.x_vals.get_or_insert_with(|| Array2::zeros((0, x.len()))); // Initialize if None
            if let Some(ref mut arr) = self.x_vals {
                arr.push_row(x.view()).unwrap(); // Push x as a new row
            }
        }
        return val;

    }

    fn gradient(&mut self, x: &Array1<f64>) -> Array1<f64> {
        let h: f64 = f64::EPSILON.sqrt() * x.iter().map(|&xi| xi.powi(2)).sum::<f64>().sqrt().max(1.0); 
        let mut grad: Array1<f64> = Array1::zeros(x.len());
        
        for i in 0..x.len() {
            let mut x_forward = x.clone();
            let mut x_backward = x.clone();

            x_forward[i] += h;
            x_backward[i] -= h;

            grad[i] = (self.value(&x_forward,true) - self.value(&x_backward,true)) / (2.0 * h);
        }
        grad
    }
    fn get_fvals(
        &mut self,
    ) -> Py<PyArray1<f64>>{
        Python::with_gil(|py| self.f_vals.get_or_insert_with(Vec::new).to_pyarray(py).to_owned())
    } 
    fn get_xvals(
        &mut self,
    ) -> Py<PyArray2<f64>>{
        Python::with_gil(|py| self.x_vals.get_or_insert_with(|| Array2::zeros((0, 0))).to_pyarray(py).to_owned())
    }

    
}
#[pyfunction]
pub fn get_function_vals(
    f: &mut Wrapper
) -> Py<PyArray1<f64>> {
    f.get_fvals()
}

#[pyfunction]
pub fn get_x_vals(
    f: &mut Wrapper
) -> Py<PyArray2<f64>> {
    f.get_xvals()
}

#[pyfunction]
pub fn minimize(
    f: &mut Wrapper,
    x:&PyArray1<f64>,
    tolerance: f64,
    max_iter: usize,
    verbose: bool
) 
->  Py<PyArray1<f64>>
{
    let mut x = x.to_owned_array();
    let mut grad_f_x: Array1<f64> = f.gradient(&x);
    let mut grad_f_xp: Array1<f64> = grad_f_x.clone();
    let mut hessian: Array2<f64> = Array2::eye(x.len());

    let mut f_x:f64 = 10.0*10e10;
    let mut f_xp:f64 = 10.0*10e10;
    let mut iter: usize = 0;
    let c_1: f64 = 10e-4;
    let c_2: f64 = 0.9;
    let iter_lim: &usize = &max_iter;
    let mut alpha: f64 = 1.0;

    while grad_f_x.norm() > tolerance && iter < max_iter {
        
        iter += 1;

        let direction: Array1<f64> = - hessian.dot(&grad_f_x);
        let mut cond_1: bool = false;
        let mut cond_2: bool = false;
        let mut iter: usize = 0;

        while !cond_1 && !cond_2 {
            alpha *= 0.5;
            
            f_x = f.value(&x, false);
            f_xp = f.value(&(&x + &(alpha * &direction)), true);

            grad_f_x = f.gradient(&x);
            grad_f_xp = f.gradient(&(&x + &(alpha * &direction)));

            cond_1 = LineSearch::armijo_rule(
                &f_xp,
                &f_x,
                &grad_f_xp,
                &direction,
                &alpha,
                &c_1
            );
            cond_2 = LineSearch::curvature_rule(
                &grad_f_xp,
                &grad_f_x,
                &direction,
                &c_2
            );
            iter += 1;
            if iter % (*iter_lim / 8) == 0 {
                if verbose{
                    println!("Line search struggling to converge, resetting hessian. Cond 1: {}, Cond 2: {}" , cond_1, cond_2);
                }
                hessian = Array2::eye(x.len());
                alpha = 1.0;
            }
            if iter == *iter_lim {
                if verbose {
                    println!("Line search failed to converge");
                }
                break;
            }

        }

        
        let next_x:Array1<f64> = &x + &(alpha * &direction);
        let dx:Array1<f64>  = &next_x - &x;
        let d_grad:Array1<f64> = &grad_f_xp - &grad_f_x;
        

        if (&next_x - &x).norm() < 10e-8 || next_x.iter().any(|&xi| xi.is_nan()) || f_xp >= f_x  {
            hessian = Array2::eye(x.len());
        } else {
            x = next_x;
            alpha = 1.0;
            hessian = update_hessian(&hessian, &dx, &d_grad);
        }

    }
    if grad_f_x.norm() < tolerance {
        println!("Optimal value of {:.*} found, termination at specified convergence of {:.*}", tolerance.log10().abs() as usize, f_x, tolerance.log10().abs() as usize, tolerance);
    }
    if iter == max_iter {
        println!("Optimal value of {:.*} found, termination at specified iteration limit of {}", tolerance.log10().abs() as usize, f_x, max_iter);
    }
    Python::with_gil(|py| x.to_pyarray(py).to_owned())
}

pub struct LineSearch;

pub trait WolfeRules {
    fn armijo_rule(f_xp: &f64,
        f_x: &f64,
        grad_f_x: &Array1<f64>,
        p_k: &Array1<f64>,
        alpha: &f64,
        c_1: &f64
    ) -> bool;

    fn curvature_rule(
        grad_f_xp: &Array1<f64>,
        grad_f_x:  &Array1<f64>,
        p_k: &Array1<f64>,
        c_2: &f64
    ) -> bool;
}

impl WolfeRules for LineSearch {
    fn armijo_rule(
        f_xp: &f64,
        f_x: &f64,
        grad_f_x: &Array1<f64>,
        p_k: &Array1<f64>,
        alpha: &f64,
        c_1: &f64
    ) -> bool {
        f_xp <=  &(f_x + alpha * c_1 * p_k.dot(grad_f_x))
    }

    fn curvature_rule(
        grad_f_xp: &Array1<f64>,
        grad_f_x: &Array1<f64>,
        p_k: &Array1<f64> , 
        c_2: &f64
    ) -> bool {
        - p_k.dot(grad_f_xp) <= - c_2 * p_k.dot(grad_f_x)
    }
}

fn update_hessian(
    hessian: &Array2<f64>,
    dx: &Array1<f64>,
    d_grad: &Array1<f64>
) -> Array2<f64> {

    let dx_dot_d_grad = dx.dot(d_grad);
    let d_grad_dot_hessian_d_grad = d_grad.dot(&hessian.dot(d_grad));
    let term1 = Array2::from_shape_fn((dx.len(), dx.len()), |(i, j)| dx[i] * dx[j] / dx_dot_d_grad);
    let hy = hessian.dot(d_grad); 
    let term2 = Array2::from_shape_fn((hy.len(), hy.len()), |(i, j)| hy[i] * hy[j] / d_grad_dot_hessian_d_grad);

    hessian + &term1 - &term2
}

#[pymodule]
fn bfgs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Wrapper>()?;
    m.add_function(wrap_pyfunction!(minimize, m)?)?;
    m.add_function(wrap_pyfunction!(get_function_vals, m)?)?;
    m.add_function(wrap_pyfunction!(get_x_vals, m)?)?;
    Ok(())
}