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
    weights: Option<DVector<f64>>,
    bias: f64,
    support_vectors: Option<DMatrix<f64>>,
    support_labels: Option<DVector<f64>>,
    support_alphas: Option<DVector<f64>>,
}

impl SupportVectorMachine {
    pub fn new(
        kernel: KernelType,
        regularization: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> Self {
        Self {
            kernel_type: kernel,
            regularization,
            error_tolerance: tolerance,
            max_iterations,
            weights: None,
            bias: 0.0,
            support_vectors: None,
            support_labels: None,
            support_alphas: None,
        }
    }

    fn kernel_function(&self, first_sample: &DVector<f64>, second_sample: &DVector<f64>) -> f64 {
        match self.kernel_type {
            KernelType::Linear => first_sample.dot(second_sample),
            KernelType::Polynomial { degree } => first_sample
                .dot(second_sample)
                .powi(i32::try_from(degree).unwrap()),
            KernelType::RBF { gamma } => {
                (-gamma * (first_sample - second_sample).norm_squared()).exp()
            }
        }
    }

    fn get_error(
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

    fn get_empirical_risk(
        &self,
        samples: &DMatrix<f64>,
        labels: &DVector<f64>,
        alphas: &DVector<f64>,
    ) -> f64 {
        let mut total_loss = 0.0;
        for i in 0..samples.nrows() {
            let mut f_i = self.bias;
            for j in 0..samples.nrows() {
                f_i += alphas[j]
                    * labels[j]
                    * self
                        .kernel_function(&samples.row(i).transpose(), &samples.row(j).transpose());
            }

            let margin = labels[i] * f_i;
            total_loss += f64::max(0.0, 1.0 - margin);
        }

        total_loss / samples.nrows() as f64
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::similar_names)]
    pub fn fit(&mut self, samples: &DMatrix<f64>, labels: &DVector<f64>) -> Vec<f64> {
        let mut alphas = DVector::zeros(samples.nrows());
        let mut risks = Vec::new();
        let mut iteration = 0;

        while iteration < self.max_iterations {
            let mut alpha_pairs_changed = 0;

            for i in 0..samples.nrows() {
                let sample_error = self.get_error(samples, labels, &alphas, i);

                if (labels[i] * sample_error < -self.error_tolerance
                    && alphas[i] < self.regularization)
                    || (labels[i] * sample_error > self.error_tolerance && alphas[i] > 0.0)
                {
                    let j = (i + 1) % samples.nrows();
                    let error_j = self.get_error(samples, labels, &alphas, j);

                    let prev_alpha_i = alphas[i];
                    let prev_alpha_j = alphas[j];

                    let (l, h) = if (labels[i] - labels[j]).abs() > f64::EPSILON {
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

                    if (alphas[j] - prev_alpha_j).abs() < 1e-5 {
                        continue;
                    }

                    alphas[i] += labels[i] * labels[j] * (prev_alpha_j - alphas[j]);

                    let first_bias_candidate = self.bias
                        - sample_error
                        - labels[i]
                            * (alphas[i] - prev_alpha_i)
                            * self.kernel_function(
                                &samples.row(i).transpose(),
                                &samples.row(i).transpose(),
                            )
                        - labels[j]
                            * (alphas[j] - prev_alpha_j)
                            * self.kernel_function(
                                &samples.row(i).transpose(),
                                &samples.row(j).transpose(),
                            );
                    let second_bias_candidate = self.bias
                        - error_j
                        - labels[i]
                            * (alphas[i] - prev_alpha_i)
                            * self.kernel_function(
                                &samples.row(i).transpose(),
                                &samples.row(j).transpose(),
                            )
                        - labels[j]
                            * (alphas[j] - prev_alpha_j)
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
                iteration += 1;
            } else {
                iteration = 0;
            }

            let risk = self.get_empirical_risk(samples, labels, &alphas);
            risks.push(risk);
        }

        let support_indices: Vec<usize> =
            (0..samples.nrows()).filter(|&i| alphas[i] > 1e-5).collect();

        self.support_vectors = Some(DMatrix::from_rows(
            &support_indices
                .iter()
                .map(|&i| samples.row(i).clone_owned())
                .collect::<Vec<_>>(),
        ));
        self.support_labels = Some(DVector::from_iterator(
            support_indices.len(),
            support_indices.iter().map(|&i| labels[i]),
        ));
        self.support_alphas = Some(DVector::from_iterator(
            support_indices.len(),
            support_indices.iter().map(|&i| alphas[i]),
        ));

        if let KernelType::Linear = self.kernel_type {
            self.weights = Some(DVector::zeros(samples.ncols()));

            for index in support_indices {
                *self.weights.as_mut().unwrap() +=
                    alphas[index] * labels[index] * samples.row(index).transpose();
            }
        }

        risks
    }

    pub fn predict(&self, features: &DVector<f64>) -> f64 {
        if let KernelType::Linear = self.kernel_type {
            let weights = self
                .weights
                .as_ref()
                .expect("Weights not set for linear kernel");

            let score = weights.dot(features) + self.bias;

            if score >= 0.0 {
                1.0
            } else {
                -1.0
            }
        } else {
            let support_vectors = self.support_vectors.as_ref().unwrap();
            let support_labels = self.support_labels.as_ref().unwrap();
            let support_alphas = self.support_alphas.as_ref().unwrap();

            let mut score = self.bias;

            for i in 0..support_vectors.nrows() {
                score += support_alphas[i]
                    * support_labels[i]
                    * self.kernel_function(&support_vectors.row(i).transpose(), features);
            }

            if score >= 0.0 {
                1.0
            } else {
                -1.0
            }
        }
    }
}
