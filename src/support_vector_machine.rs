use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone, Copy)]
pub enum KernelType {
    Linear,
    Polynomial { degree: u32 },
    RBF { gamma: f64 },
}

pub struct SVM {
    kernel: KernelType,
    learning_rate: f64,
    regularization: f64,
    tolerance: f64,
    max_iterations: usize,
    weights: DVector<f64>,
    bias: f64,
}

impl SVM {
    pub fn new(
        kernel: KernelType,
        learning_rate: f64,
        regularization: f64,
        tolerance: f64,
        max_iterations: usize,
        feature_dimensions: usize,
    ) -> Self {
        Self {
            kernel,
            learning_rate,
            regularization,
            tolerance,
            max_iterations,
            weights: DVector::zeros(feature_dimensions),
            bias: 0.0,
        }
    }

    fn kernel_function(&self, x: &DVector<f64>, y: &DVector<f64>) -> f64 {
        match self.kernel {
            KernelType::Linear => x.dot(y),
            KernelType::Polynomial { degree } => x.dot(y).powi(degree as i32),
            KernelType::RBF { gamma } => (-gamma * (x - y).norm_squared()).exp(),
        }
    }

    fn compute_gradient(
        &self,
        samples: &DMatrix<f64>,
        labels: &DVector<f64>,
        alpha: &DVector<f64>,
    ) -> (DVector<f64>, f64) {
        let mut weights = DVector::zeros(samples.ncols());
        let mut bias = 0.0;

        for i in 0..samples.nrows() {
            let label_i = labels[i];
            let alpha_i = alpha[i];
            let sample_i = samples.row(i).transpose();
            let mut sum = 0.0;

            for j in 0..samples.nrows() {
                let label_j = labels[j];
                let alpha_j = alpha[j];
                let sample_j = samples.row(j).transpose();
                sum += alpha_j * label_j * self.kernel_function(&sample_i, &sample_j);
            }

            weights += alpha_i * label_i * sample_i;
            bias += alpha_i * (label_i - sum);
        }

        (weights, bias / labels.len() as f64)
    }

    pub fn fit(&mut self, samples: &DMatrix<f64>, labels: &DVector<f64>) {
        let mut alpha = DVector::zeros(samples.nrows());
        let mut iter = 0;

        while iter < self.max_iterations {
            let (weights, bias) = self.compute_gradient(samples, labels, &alpha);
            self.weights = weights;
            self.bias = bias;

            for i in 0..samples.nrows() {
                let margin =
                    labels[i] * (self.weights.dot(&samples.row(i).transpose()) + self.bias);
                let new_alpha = alpha[i] + self.learning_rate * (1.0 - margin);
                alpha[i] = new_alpha.clamp(0.0, self.regularization);
            }

            iter += 1;
            if alpha.iter().all(|&a| a.abs() < self.tolerance) {
                break;
            }
        }
    }

    pub fn predict(&self, sample: &DVector<f64>) -> f64 {
        let score = self.weights.dot(sample) + self.bias;
        if score >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }
}
