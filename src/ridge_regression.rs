use nalgebra::DMatrix;
use ndarray::{Array1, Array2};

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

    const FEATURES_WITH_BIAS_DIMENSION: usize = DIMENSIONS + 1;

    pub fn fit(&mut self, samples: &[Sample]) {
        let samples_count = samples.len();

        // X
        let mut features = Array2::zeros((samples_count, Self::FEATURES_WITH_BIAS_DIMENSION));

        // y
        let mut labels = Array1::zeros(samples_count);

        for (i, sample) in samples.iter().enumerate() {
            features[(i, 0)] = 1.0;

            for (j, &feature) in sample.features.iter().enumerate() {
                features[(i, j + 1)] = feature;
            }

            labels[i] = sample.label;
        }

        let features_transpose = features.t();

        // (tau * I)
        let mut regularization: Array2<f64> =
            Array2::eye(Self::FEATURES_WITH_BIAS_DIMENSION) * self.regularization;
        regularization[(0, 0)] = 0.0;

        // (X^T * X + tau * I)
        let covariance = features_transpose.dot(&features) + regularization;

        let covariance = DMatrix::from_vec(
            Self::FEATURES_WITH_BIAS_DIMENSION,
            Self::FEATURES_WITH_BIAS_DIMENSION,
            covariance.iter().copied().collect(),
        );
        let features_transpose = DMatrix::from_vec(
            Self::FEATURES_WITH_BIAS_DIMENSION,
            samples_count,
            features_transpose.iter().copied().collect(),
        );
        let label_vector = DMatrix::from_column_slice(samples_count, 1, labels.as_slice().unwrap());

        // (X^T * X + tau * I)^-1
        if let Some(covariance_inverse) = covariance.try_inverse() {
            // X^T * y
            let features_targets_product = features_transpose * label_vector;

            // (X^T * X + tau * I)^-1 * (X^T * y)
            let weights = covariance_inverse * features_targets_product;

            self.weights = Array1::from(weights.column(0).as_slice().to_vec());
        }
    }

    pub fn predict(&self, features: &[f64; DIMENSIONS]) -> f64 {
        let mut extended_features = Array1::zeros(Self::FEATURES_WITH_BIAS_DIMENSION);
        extended_features[0] = 1.0;
        for (i, &feature) in features.iter().enumerate() {
            extended_features[i + 1] = feature;
        }

        let prediction = extended_features.dot(&self.weights);

        if prediction > 0.0 {
            1.0
        } else {
            -1.0
        }
    }
}
