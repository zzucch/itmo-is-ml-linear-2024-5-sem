use crate::parse::{Sample, DIMENSIONS};
use ndarray::{Array1, ArrayView1};

#[derive(Debug, Clone, Copy)]
pub enum LossType {
    Logistic,
    Exponential,
    Hinge,
}

pub struct LinearClassifier {
    pub weights: Array1<f64>,
    pub learning_rate: f64,
    pub elastic_net_regularization: f64,
    pub loss_type: LossType,
}

impl LinearClassifier {
    pub fn new(learning_rate: f64, elastic_net_regularization: f64, loss_type: LossType) -> Self {
        Self {
            weights: Array1::zeros(DIMENSIONS),
            learning_rate,
            elastic_net_regularization,
            loss_type,
        }
    }

    pub fn fit(&mut self, samples: &[Sample], number_of_epochs: usize) -> Vec<f64> {
        let mut risks = Vec::with_capacity(number_of_epochs);

        for _ in 0..number_of_epochs {
            self.step(samples);

            let empirical_risk = self.get_empirical_risk(samples);
            risks.push(empirical_risk);
        }

        risks
    }

    pub fn predict(&self, features: &[f64; DIMENSIONS]) -> f64 {
        let dot_product = ArrayView1::from(features).dot(&self.weights);
        let predicted_value = 1.0 / (1.0 + (-dot_product).exp());

        if predicted_value > 0.5 {
            1.0
        } else {
            -1.0
        }
    }

    fn step(&mut self, samples: &[Sample]) {
        let mut gradient = self.get_loss_gradient(samples);

        gradient += &self.get_elastic_net_regularization_gradient();

        self.weights = &self.weights - self.learning_rate * gradient;
    }

    fn get_elastic_net_regularization_gradient(&self) -> Array1<f64> {
        let l1_term = self.weights.mapv(f64::signum);
        let l2_term = self.weights.clone();

        self.elastic_net_regularization * (l1_term + 2.0 * l2_term)
    }

    fn get_empirical_risk(&self, samples: &[Sample]) -> f64 {
        let n_samples = samples.len() as f64;
        let mut total_loss = 0.0;

        for sample in samples {
            let dot_product = ArrayView1::from(&sample.features).dot(&self.weights);
            let margin = sample.label * dot_product;

            let loss = match self.loss_type {
                LossType::Logistic => (1.0 / (1.0 + (-margin).exp())).ln(),
                LossType::Exponential => (-margin).exp(),
                LossType::Hinge => (1.0 - margin).max(0.0),
            };

            total_loss += loss;
        }

        total_loss / n_samples
    }

    fn get_loss_gradient(&self, samples: &[Sample]) -> Array1<f64> {
        match self.loss_type {
            LossType::Logistic => self.get_logistic_loss_gradient(samples),
            LossType::Exponential => self.get_exponential_loss_gradient(samples),
            LossType::Hinge => self.get_hinge_loss_gradient(samples),
        }
    }

    fn get_logistic_loss_gradient(&self, samples: &[Sample]) -> Array1<f64> {
        let mut gradient = Array1::zeros(DIMENSIONS);
        let n_samples = samples.len() as f64;

        for sample in samples {
            let dot_product = ArrayView1::from(&sample.features).dot(&self.weights);
            let margin = -sample.label * dot_product;
            let probability = 1.0 / (1.0 + margin.exp());

            let sample_features = ArrayView1::from(&sample.features);
            gradient.zip_mut_with(&sample_features, |current_gradient, &feature_value| {
                *current_gradient += feature_value * sample.label * (1.0 - probability);
            });
        }

        gradient / n_samples
    }

    fn get_exponential_loss_gradient(&self, samples: &[Sample]) -> Array1<f64> {
        let mut gradient = Array1::zeros(DIMENSIONS);
        let n_samples = samples.len() as f64;

        for sample in samples {
            let dot_product = ArrayView1::from(&sample.features).dot(&self.weights);
            let margin = sample.label * dot_product;
            let exp_term = (-margin).exp();

            let sample_features = ArrayView1::from(&sample.features);
            gradient.zip_mut_with(&sample_features, |current_gradient, &feature_value| {
                *current_gradient += -sample.label * feature_value * exp_term;
            });
        }

        gradient / n_samples
    }

    fn get_hinge_loss_gradient(&self, samples: &[Sample]) -> Array1<f64> {
        let mut gradient = Array1::zeros(DIMENSIONS);

        for sample in samples {
            let dot_product = ArrayView1::from(&sample.features).dot(&self.weights);
            let margin = sample.label * dot_product;

            if margin < 1.0 {
                let sample_features = ArrayView1::from(&sample.features);
                gradient.zip_mut_with(&sample_features, |current_gradient, &feature_value| {
                    *current_gradient += -sample.label * feature_value;
                });
            }
        }

        gradient
    }
}
