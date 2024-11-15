use linear::{
    gradient_descent::{LinearClassifier, LossType},
    parse::{csv_entries_to_ridge_samples, Sample, DIMENSIONS},
    ridge_regression::RidgeRegression,
};

fn split(samples: &[Sample], train_ratio: f64) -> (Vec<Sample>, Vec<Sample>) {
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    let train_size = (samples.len() as f64 * train_ratio) as usize;
    let (first, second) = samples.split_at(train_size);

    (first.to_vec(), second.to_vec())
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

    Ok(())
}
