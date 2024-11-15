use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone, Copy)]
pub enum KernelType {
    Linear,
    Polynomial { degree: u32 },
    RBF { gamma: f64 },
}

pub struct SupportVectorMachine {
    kernel_type: KernelType,
    regularization: f64,
    error_tolerance: f64,
    max_iterations: usize,
    weights: DVector<f64>,
    bias: f64,
}

impl SupportVectorMachine {
    pub fn new(
        kernel: KernelType,
        regularization: f64,
        tolerance: f64,
        max_iterations: usize,
        feature_dimensions: usize,
    ) -> Self {
        Self {
            kernel_type: kernel,
            regularization,
            error_tolerance: tolerance,
            max_iterations,
            weights: DVector::zeros(feature_dimensions),
            bias: 0.0,
        }
    }

    fn kernel_function(&self, first_sample: &DVector<f64>, second_sample: &DVector<f64>) -> f64 {
        match self.kernel_type {
            KernelType::Linear => first_sample.dot(second_sample),
            KernelType::Polynomial { degree } => {
                first_sample.dot(second_sample).powi(degree as i32)
            }
            KernelType::RBF { gamma } => {
                (-gamma * (first_sample - second_sample).norm_squared()).exp()
            }
        }
    }

    fn compute_error(
        &self,
        samples: &DMatrix<f64>,
        labels: &DVector<f64>,
        alphas: &DVector<f64>,
        i: usize,
    ) -> f64 {
        let mut f_i = self.bias;
        for j in 0..samples.nrows() {
            f_i += alphas[j]
                * labels[j]
                * self.kernel_function(&samples.row(i).transpose(), &samples.row(j).transpose());
        }
        f_i - labels[i]
    }

    pub fn fit(&mut self, samples: &DMatrix<f64>, labels: &DVector<f64>) {
        let mut alphas = DVector::zeros(samples.nrows());
        let mut iter = 0;

        while iter < self.max_iterations {
            let mut alpha_pairs_changed = 0;

            for i in 0..samples.nrows() {
                let sample_error = self.compute_error(samples, labels, &alphas, i);

                if (labels[i] * sample_error < -self.error_tolerance
                    && alphas[i] < self.regularization)
                    || (labels[i] * sample_error > self.error_tolerance && alphas[i] > 0.0)
                {
                    let j = (i + 1) % samples.nrows();
                    let error_j = self.compute_error(samples, labels, &alphas, j);

                    let alpha_i_old = alphas[i];
                    let alpha_j_old = alphas[j];

                    let (l, h) = if labels[i] != labels[j] {
                        (
                            f64::max(0.0, alphas[j] - alphas[i]),
                            f64::min(
                                self.regularization,
                                self.regularization + alphas[j] - alphas[i],
                            ),
                        )
                    } else {
                        (
                            f64::max(0.0, alphas[i] + alphas[j] - self.regularization),
                            f64::min(self.regularization, alphas[i] + alphas[j]),
                        )
                    };

                    if l >= h {
                        continue;
                    }

                    let eta =
                        2.0 * self.kernel_function(
                            &samples.row(i).transpose(),
                            &samples.row(j).transpose(),
                        ) - self.kernel_function(
                            &samples.row(i).transpose(),
                            &samples.row(i).transpose(),
                        ) - self.kernel_function(
                            &samples.row(j).transpose(),
                            &samples.row(j).transpose(),
                        );

                    if eta >= 0.0 {
                        continue;
                    }

                    alphas[j] -= labels[j] * (sample_error - error_j) / eta;
                    alphas[j] = alphas[j].clamp(l, h);

                    if (alphas[j] - alpha_j_old).abs() < 1e-5 {
                        continue;
                    }

                    alphas[i] += labels[i] * labels[j] * (alpha_j_old - alphas[j]);

                    let first_bias_candidate = self.bias
                        - sample_error
                        - labels[i]
                            * (alphas[i] - alpha_i_old)
                            * self.kernel_function(
                                &samples.row(i).transpose(),
                                &samples.row(i).transpose(),
                            )
                        - labels[j]
                            * (alphas[j] - alpha_j_old)
                            * self.kernel_function(
                                &samples.row(i).transpose(),
                                &samples.row(j).transpose(),
                            );
                    let second_bias_candidate = self.bias
                        - error_j
                        - labels[i]
                            * (alphas[i] - alpha_i_old)
                            * self.kernel_function(
                                &samples.row(i).transpose(),
                                &samples.row(j).transpose(),
                            )
                        - labels[j]
                            * (alphas[j] - alpha_j_old)
                            * self.kernel_function(
                                &samples.row(j).transpose(),
                                &samples.row(j).transpose(),
                            );

                    if 0.0 < alphas[i] && alphas[i] < self.regularization {
                        self.bias = first_bias_candidate;
                    } else if 0.0 < alphas[j] && alphas[j] < self.regularization {
                        self.bias = second_bias_candidate;
                    } else {
                        self.bias = (first_bias_candidate + second_bias_candidate) / 2.0;
                    }

                    alpha_pairs_changed += 1;
                }
            }

            if alpha_pairs_changed == 0 {
                iter += 1;
            } else {
                iter = 0;
            }
        }

        self.weights = DVector::zeros(samples.ncols());
        for i in 0..samples.nrows() {
            if alphas[i] > 0.0 {
                self.weights += alphas[i] * labels[i] * samples.row(i).transpose();
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
