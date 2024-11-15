use nalgebra::DMatrix;
use ndarray::{Array1, Array2, ArrayView1};

use crate::parse::{Sample, DIMENSIONS};
#[derive(Default)]
pub struct RidgeRegression {
    weights: Array1<f64>,
    regularization: f64,
}

impl RidgeRegression {
    pub fn new(regularization: f64) -> Self {
        Self {
            weights: Array1::zeros(DIMENSIONS),
            regularization,
        }
    }

    pub fn fit(&mut self, samples: &[Sample]) {
        let samples_count = samples.len();

        // X
        let mut features = Array2::zeros((samples_count, DIMENSIONS));

        // y
        let mut labels = Array1::zeros(samples_count);

        for (i, sample) in samples.iter().enumerate() {
            features
                .row_mut(i)
                .assign(&ArrayView1::from(&sample.features));
            labels[i] = sample.label;
        }

        let features_transpose = features.t();

        // (tau * I)
        let regularization: Array2<f64> = Array2::eye(DIMENSIONS) * self.regularization;

        // (X^T * X + tau * I)
        let covariance = features_transpose.dot(&features) + regularization;

        let covariance =
            DMatrix::from_vec(DIMENSIONS, DIMENSIONS, covariance.iter().cloned().collect());
        let feature_transpose = DMatrix::from_vec(
            DIMENSIONS,
            samples_count,
            features_transpose.iter().cloned().collect(),
        );
        let label_vector = DMatrix::from_column_slice(samples_count, 1, labels.as_slice().unwrap());

        // (X^T * X + tau * I)^-1
        if let Some(covariance_inverse) = covariance.try_inverse() {
            // X^T * y
            let features_targets_product = feature_transpose * label_vector;

            // (X^T * X + tau * I)^-1 * (X^T * y)
            let weights = covariance_inverse * features_targets_product;

            self.weights = Array1::from(weights.column(0).as_slice().to_vec());
        }
    }

    pub fn predict(&self, features: &[f64; DIMENSIONS]) -> f64 {
        let prediction = ArrayView1::from(features).dot(&self.weights);

        if prediction > 0.0 {
            1.0
        } else {
            -1.0
        }
    }
}
