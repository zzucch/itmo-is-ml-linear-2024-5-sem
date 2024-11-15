use linear::{
    gradient_descent::{LinearClassifier, LossType},
    parse::{csv_entries_to_ridge_samples, Sample, DIMENSIONS},
    ridge_regression::RidgeRegression,
    support_vector_machine::{KernelType, SupportVectorMachine},
};
use nalgebra::{DMatrix, DVector};

fn split(samples: &[Sample], train_ratio: f64) -> (Vec<Sample>, Vec<Sample>) {
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    let train_size = (samples.len() as f64 * train_ratio) as usize;
    let (first, second) = samples.split_at(train_size);

    (first.to_vec(), second.to_vec())
}

fn convert_samples_to_matrix(samples: &[Sample]) -> DMatrix<f64> {
    let data: Vec<f64> = samples.iter().flat_map(|s| s.features.to_vec()).collect();

    DMatrix::from_vec(samples.len(), samples[0].features.len(), data)
}

fn convert_labels_to_vector(samples: &[Sample]) -> DVector<f64> {
    DVector::from_vec(samples.iter().map(|s| s.label).collect::<Vec<f64>>())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const DATA_FILEPATH: &str = "data/breast-cancer.csv";

    let entries = linear::parse::parse(DATA_FILEPATH)?;
    assert!(!entries.is_empty());
    assert_eq!(entries.first().unwrap().values.len(), DIMENSIONS);

    let samples = csv_entries_to_ridge_samples(entries);

    const TRAIN_RATIO: f64 = 0.6;
    const VALIDATION_RATIO: f64 = 0.6; // of samples that are not train

    let (train_samples, test_samples) = split(&samples, TRAIN_RATIO);
    let (test_samples, _validation_samples) = split(&test_samples, VALIDATION_RATIO);

    pub const REGULARIZATION: f64 = 10.0;

    let mut model = RidgeRegression::new(REGULARIZATION);
    model.fit(&train_samples);

    let mut correct_predictions = 0;
    for sample in &test_samples {
        let prediction = model.predict(&sample.features);

        #[allow(clippy::float_cmp)]
        if prediction == sample.label {
            correct_predictions += 1;
        }
    }

    let accuracy = (correct_predictions as f64 / test_samples.len() as f64) * 100.0;
    println!("ridge regression accuracy: {accuracy:.3}%");

    const ELASTIC_NET_REGULARIZATION: f64 = 0.01;
    const LEARNING_RATE: f64 = 0.01;
    const EPOCHS: usize = 1000;

    for loss_type in [LossType::Logistic, LossType::Exponential, LossType::Hinge] {
        let mut model = LinearClassifier::new(LEARNING_RATE, ELASTIC_NET_REGULARIZATION, loss_type);

        model.fit(&train_samples, EPOCHS);

        let mut correct_predictions = 0;
        for sample in &test_samples {
            let prediction = model.predict(&sample.features);

            #[allow(clippy::float_cmp)]
            if prediction == sample.label {
                correct_predictions += 1;
            }
        }

        let accuracy = (correct_predictions as f64 / test_samples.len() as f64) * 100.0;
        println!("{loss_type:?} regression accuracy: {accuracy:.3}%");
    }

    let train_matrix = convert_samples_to_matrix(&train_samples);
    let train_labels = convert_labels_to_vector(&train_samples);
    let test_matrix = convert_samples_to_matrix(&test_samples);
    let test_labels = convert_labels_to_vector(&test_samples);

    const KERNEL: KernelType = KernelType::Linear;
    const SVM_REGULARIZATION: f64 = 0.1;
    const TOLERANCE: f64 = 0.001;
    const MAX_ITERATIONS: usize = 10;

    let mut svm_model = SupportVectorMachine::new(
        KERNEL,
        SVM_REGULARIZATION,
        TOLERANCE,
        MAX_ITERATIONS,
        DIMENSIONS,
    );
    svm_model.fit(&train_matrix, &train_labels);

    let mut correct_predictions = 0;
    for i in 0..test_matrix.nrows() {
        let sample = test_matrix.row(i).transpose();
        let prediction = svm_model.predict(&sample);

        #[allow(clippy::float_cmp)]
        if prediction == test_labels[i] {
            correct_predictions += 1;
        }
    }

    let accuracy = (correct_predictions as f64 / test_samples.len() as f64) * 100.0;
    println!("SVM regression accuracy: {accuracy:.3}%");

    Ok(())
}
