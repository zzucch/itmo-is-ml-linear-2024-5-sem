use linear::ridge_regression::{RidgeRegression, Sample};

fn csv_entries_to_ridge_samples(entries: Vec<linear::parse::CsvEntry>) -> Vec<Sample> {
    entries
        .into_iter()
        .map(|entry| Sample {
            features: entry.values.try_into().unwrap(),
            label: match entry.diagnosis {
                linear::parse::Diagnosis::Malignant => 1.0,
                linear::parse::Diagnosis::Benign => -1.0,
            },
        })
        .collect()
}

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
    assert_eq!(
        entries.first().unwrap().values.len(),
        linear::ridge_regression::DIMENSIONS
    );

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
        if prediction == sample.label {
            correct_predictions += 1;
        }
    }

    let accuracy = (correct_predictions as f64 / test_samples.len() as f64) * 100.0;
    println!("ridge regression accuracy: {:.3}%", accuracy);

    Ok(())
}
