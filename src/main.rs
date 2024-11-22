use linear::{
    gradient_descent::{LinearClassifier, LossType},
    parse::{csv_entries_to_ridge_samples, Sample, DIMENSIONS},
    ridge_regression::RidgeRegression,
    support_vector_machine::{KernelType, SupportVectorMachine},
};
use nalgebra::{DMatrix, DVector};
use plotters::{
    chart::ChartBuilder,
    prelude::{BitMapBackend, IntoDrawingArea, PathElement},
    series::LineSeries,
    style::{Color, RED, WHITE},
};
use std::path::Path;

fn split(samples: &[Sample], train_ratio: f64) -> (Vec<Sample>, Vec<Sample>) {
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    let train_size = (samples.len() as f64 * train_ratio) as usize;
    let (first, second) = samples.split_at(train_size);

    (first.to_vec(), second.to_vec())
}

fn plot_learning_curve(
    risks: &[f64],
    title: &str,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(Path::new(filename), (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_risk = risks.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_risk = risks.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let margin = 0.1 * (max_risk - min_risk);

    let y_range = (min_risk - margin)..(max_risk + margin);
    let x_range = 0..risks.len() as i32;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            risks.iter().enumerate().map(|(i, &r)| (i as i32, r)),
            &RED,
        ))?
        .label("Empirical Risk")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .draw()?;

    Ok(())
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

    run_ridge_regression(&train_samples, &test_samples);

    run_linear_classification(&train_samples, &test_samples);

    let train_matrix = convert_samples_to_matrix(&train_samples);
    let train_labels = convert_labels_to_vector(&train_samples);
    let test_matrix = convert_samples_to_matrix(&test_samples);
    let test_labels = convert_labels_to_vector(&test_samples);

    run_svm(&train_matrix, &train_labels, &test_matrix, &test_labels);

    learning_curve_linear_classification(&train_samples, "learning_curve_linear_classification");
    learning_curve_svm(&train_matrix, &train_labels, "learning_curve_svm");

    Ok(())
}

fn run_ridge_regression(train_samples: &[Sample], test_samples: &[Sample]) {
    pub const REGULARIZATION: f64 = 10.0;

    let mut model = RidgeRegression::new(REGULARIZATION);
    model.fit(&train_samples);

    let mut correct_predictions = 0;
    for sample in test_samples {
        let prediction = model.predict(&sample.features);

        #[allow(clippy::float_cmp)]
        if prediction == sample.label {
            correct_predictions += 1;
        }
    }

    let accuracy = (correct_predictions as f64 / test_samples.len() as f64) * 100.0;
    println!("Ridge regression accuracy: {accuracy:.3}%");
}

fn run_linear_classification(train_samples: &[Sample], test_samples: &[Sample]) {
    const ELASTIC_NET_REGULARIZATION: f64 = 0.01;
    const LEARNING_RATE: f64 = 0.01;
    const EPOCHS: usize = 1000;

    for loss_type in [LossType::Logistic, LossType::Exponential, LossType::Hinge] {
        let mut model = LinearClassifier::new(LEARNING_RATE, ELASTIC_NET_REGULARIZATION, loss_type);

        model.fit(&train_samples, EPOCHS);

        let mut correct_predictions = 0;
        for sample in test_samples {
            let prediction = model.predict(&sample.features);

            #[allow(clippy::float_cmp)]
            if prediction == sample.label {
                correct_predictions += 1;
            }
        }

        let accuracy = (correct_predictions as f64 / test_samples.len() as f64) * 100.0;
        println!("{loss_type:?} regression accuracy: {accuracy:.3}%");
    }
}

fn run_svm(
    train_matrix: &DMatrix<f64>,
    train_labels: &DVector<f64>,
    test_matrix: &DMatrix<f64>,
    test_labels: &DVector<f64>,
) {
    const KERNEL: KernelType = KernelType::RBF { gamma: 0.5 };
    const SVM_REGULARIZATION: f64 = 0.1;
    const TOLERANCE: f64 = 0.01;
    const MAX_ITERATIONS: usize = 2;

    let mut svm_model =
        SupportVectorMachine::new(KERNEL, SVM_REGULARIZATION, TOLERANCE, MAX_ITERATIONS);
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

    let accuracy = (correct_predictions as f64 / test_matrix.nrows() as f64) * 100.0;
    println!("SVM regression accuracy: {accuracy:.3}%");
}

fn learning_curve_linear_classification(train_samples: &[Sample], filename: &str) {
    const ELASTIC_NET_REGULARIZATION: f64 = 0.01;
    const LEARNING_RATE: f64 = 0.01;
    const EPOCHS: usize = 1000;

    for loss_type in [LossType::Logistic, LossType::Exponential, LossType::Hinge] {
        let mut model = LinearClassifier::new(LEARNING_RATE, ELASTIC_NET_REGULARIZATION, loss_type);

        let risks = model.fit(&train_samples, EPOCHS);

        let name = &format!("{}_{:?}", filename, loss_type);
        plot_learning_curve(&risks, name, &format!("{}.png", name)).unwrap();
    }
}

fn learning_curve_svm(train_matrix: &DMatrix<f64>, train_labels: &DVector<f64>, filename: &str) {
    const KERNEL: KernelType = KernelType::RBF { gamma: 0.5 };
    const SVM_REGULARIZATION: f64 = 0.1;
    const TOLERANCE: f64 = 0.01;
    const MAX_ITERATIONS: usize = 2;

    let mut svm_model =
        SupportVectorMachine::new(KERNEL, SVM_REGULARIZATION, TOLERANCE, MAX_ITERATIONS);

    let risks = svm_model.fit(&train_matrix, &train_labels);

    plot_learning_curve(&risks, "learning_curve_svm", &format!("{}.png", filename)).unwrap();
}
