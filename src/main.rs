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
    style::{Color, BLUE, GREEN, RED, WHITE},
};
use std::{error::Error, path::Path};

fn split(samples: &[Sample], train_ratio: f64) -> (Vec<Sample>, Vec<Sample>) {
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    let train_size = (samples.len() as f64 * train_ratio) as usize;

    let (first, second) = samples.split_at(train_size);
    (first.to_vec(), second.to_vec())
}

fn get_std(values: &[f64]) -> f64 {
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;

    variance.sqrt()
}

fn calculate_f1_score(samples: &[Sample], predictions: &[f64]) -> f64 {
    let mut true_positive_count = 0;
    let mut false_positive_count = 0;
    let mut false_negative_count = 0;

    for (actual, predicted) in samples.iter().zip(predictions.iter()) {
        if (actual.label - *predicted).abs() < f64::EPSILON {
            true_positive_count += 1;
        } else {
            match predicted {
                // malignant
                1.0 => {
                    false_positive_count += 1;
                }
                // benign
                -1.0 => {
                    false_negative_count += 1;
                }
                _ => unreachable!(),
            }
        }
    }

    let precision = if true_positive_count + false_positive_count > 0 {
        true_positive_count as f64 / (true_positive_count + false_positive_count) as f64
    } else {
        0.0
    };
    let recall = if true_positive_count + false_negative_count > 0 {
        true_positive_count as f64 / (true_positive_count + false_negative_count) as f64
    } else {
        0.0
    };

    if precision + recall > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else {
        0.0
    }
}

fn plot_learning_curve(
    values: &[(i32, f64)],
    title: &str,
    label: &str,
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(Path::new(filename), (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_value = values.iter().map(|&(_, y)| y).fold(f64::INFINITY, f64::min);
    let max_value = values
        .iter()
        .map(|&(_, y)| y)
        .fold(f64::NEG_INFINITY, f64::max);
    let margin = 0.1 * (max_value - min_value);

    #[allow(clippy::range_plus_one)]
    let x_range = values[0].0..values.last().unwrap().0 + 1;
    let y_range = (min_value - margin)..(max_value + margin);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(values.iter().copied(), &RED))?
        .label(label)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], RED));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .draw()?;

    println!("plot saved to {filename}");

    Ok(())
}

fn plot_learning_curve_with_confidence_intervals(
    values: &[(i32, f64)],
    title: &str,
    label: &str,
    filename: &str,
    confidence_intervals: &[f64],
    ridge_regression_f1_score: f64,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(Path::new(filename), (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_range = values[0].0 as f64..values.last().unwrap().0 as f64 + 1.0;
    let y_range = 0.0..1.0;

    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_possible_truncation)]
    let x_values: Vec<f64> = (x_range.start as usize..x_range.end as usize)
        .map(|x| x as f64)
        .collect();

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh().draw()?;

    let filled_area = {
        let mut area = Vec::new();
        for (&(x, y), &high) in values.iter().zip(confidence_intervals.iter()) {
            area.push((x as f64, y + high));
        }
        for (&(x, y), &low) in values.iter().zip(confidence_intervals.iter()).rev() {
            area.push((x as f64, y - low));
        }
        area
    };

    chart.draw_series(std::iter::once(plotters::prelude::Polygon::new(
        filled_area,
        BLUE.mix(0.2).filled(),
    )))?;

    chart
        .draw_series(LineSeries::new(
            x_values.iter().map(|&x| (x, ridge_regression_f1_score)),
            GREEN,
        ))?
        .label("Ridge Regression F1 Score")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], GREEN));

    chart
        .draw_series(LineSeries::new(
            values.iter().map(|&(x, y)| (x as f64, y)),
            &RED,
        ))?
        .label(label)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], RED));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .draw()?;

    println!("plot saved to {filename}");

    Ok(())
}

fn convert_samples_to_matrix(samples: &[Sample]) -> DMatrix<f64> {
    let data: Vec<f64> = samples
        .iter()
        .flat_map(|sample| sample.features.to_vec())
        .collect();

    DMatrix::from_vec(samples.len(), samples[0].features.len(), data)
}

fn convert_labels_to_vector(samples: &[Sample]) -> DVector<f64> {
    DVector::from_vec(
        samples
            .iter()
            .map(|sample| sample.label)
            .collect::<Vec<f64>>(),
    )
}

fn main() -> Result<(), Box<dyn Error>> {
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

    run_asymptotically_equal(&train_samples, &train_matrix, &train_labels);

    run_train_learning_curve_linear_classification(
        &train_samples,
        "train_learning_curve_linear_classification",
    );
    run_train_learning_curve_svm(&train_matrix, &train_labels, "train_learning_curve_svm");

    learning_curve_test_linear_classification(
        &samples,
        "test_learning_curve_linear_classification",
    );
    learning_curve_test_svm(&samples, "test_learning_curve_svm");

    Ok(())
}

fn run_ridge_regression(train_samples: &[Sample], test_samples: &[Sample]) {
    pub const REGULARIZATION: f64 = 0.0005;

    let mut model = RidgeRegression::new(REGULARIZATION);
    model.fit(train_samples);

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

        model.fit(train_samples, EPOCHS);

        let mut correct_predictions = 0;
        for sample in test_samples {
            let prediction = model.predict(&sample.features);

            #[allow(clippy::float_cmp)]
            if prediction == sample.label {
                correct_predictions += 1;
            }
        }

        let accuracy = (correct_predictions as f64 / test_samples.len() as f64) * 100.0;
        println!("{loss_type:?} linear classification accuracy: {accuracy:.3}%");
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
    svm_model.fit(train_matrix, train_labels);

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
    println!("SVM classification accuracy: {accuracy:.3}%");
}

fn run_asymptotically_equal(
    train_samples: &[Sample],
    train_matrix: &DMatrix<f64>,
    train_labels: &DVector<f64>,
) {
    const LOSS_TYPE: LossType = LossType::Logistic;
    const ELASTIC_NET_REGULARIZATION: f64 = 0.01;
    const LEARNING_RATE: f64 = 0.01;

    const KERNEL: KernelType = KernelType::RBF { gamma: 0.5 };
    const SVM_REGULARIZATION: f64 = 0.1;
    const TOLERANCE: f64 = 0.01;

    const COEF: usize = 3;
    let size = train_samples.len();
    let epoch_count = COEF * size;
    let max_iterations = COEF;

    let mut model = LinearClassifier::new(LEARNING_RATE, ELASTIC_NET_REGULARIZATION, LOSS_TYPE);

    model.fit(train_samples, epoch_count);

    let mut svm_model =
        SupportVectorMachine::new(KERNEL, SVM_REGULARIZATION, TOLERANCE, max_iterations);
    svm_model.fit(train_matrix, train_labels);
}

fn run_train_learning_curve_linear_classification(train_samples: &[Sample], filename: &str) {
    const LEARNING_RATE: f64 = 0.01;
    const ELASTIC_NET_REGULARIZATION: f64 = 0.01;
    const LOSS_TYPE: LossType = LossType::Exponential;
    const EPOCHS: usize = 1000;

    let mut model = LinearClassifier::new(LEARNING_RATE, ELASTIC_NET_REGULARIZATION, LOSS_TYPE);

    let risks = model.fit(train_samples, EPOCHS);

    let risk_with_epochs: Vec<(i32, f64)> = risks
        .into_iter()
        .enumerate()
        .map(|(epoch, risk)| (i32::try_from(epoch).unwrap(), risk))
        .collect();

    plot_learning_curve(
        &risk_with_epochs,
        filename,
        "Empirical risk",
        &format!("{filename}.png"),
    )
    .unwrap();
}

fn run_train_learning_curve_svm(
    train_matrix: &DMatrix<f64>,
    train_labels: &DVector<f64>,
    filename: &str,
) {
    const KERNEL: KernelType = KernelType::RBF { gamma: 0.5 };
    const SVM_REGULARIZATION: f64 = 0.1;
    const TOLERANCE: f64 = 0.01;
    const MAX_ITERATIONS: usize = 10;

    let mut svm_model =
        SupportVectorMachine::new(KERNEL, SVM_REGULARIZATION, TOLERANCE, MAX_ITERATIONS);

    let risks = svm_model.fit(train_matrix, train_labels);

    let risk_with_epochs: Vec<(i32, f64)> = risks
        .into_iter()
        .enumerate()
        .map(|(epoch, risk)| (i32::try_from(epoch).unwrap(), risk))
        .collect();

    plot_learning_curve(
        &risk_with_epochs,
        "learning_curve_svm",
        "Empirical risk",
        &format!("{filename}.png"),
    )
    .unwrap();
}

fn learning_curve_test_linear_classification(samples: &[Sample], filename: &str) {
    const LEARNING_RATE: f64 = 0.01;
    const ELASTIC_NET_REGULARIZATION: f64 = 0.01;
    const LOSS_TYPE: LossType = LossType::Exponential;
    const EPOCHS: usize = 1000;

    let mut test_f1s = vec![];

    let ratio_percents = 1..=95;

    for ratio_percent in ratio_percents.clone() {
        let ratio = ratio_percent as f64 / 100.0;
        let (train_samples, test_samples) = split(samples, ratio);

        let mut model = LinearClassifier::new(LEARNING_RATE, ELASTIC_NET_REGULARIZATION, LOSS_TYPE);
        model.fit(&train_samples, EPOCHS);

        let mut test_predictions = Vec::with_capacity(test_samples.len());

        for sample in &test_samples {
            let prediction = model.predict(&sample.features);
            test_predictions.push(prediction);
        }

        let test_f1 = calculate_f1_score(&test_samples, &test_predictions);
        test_f1s.push(test_f1);
    }

    let std = get_std(&test_f1s);

    let f1_with_ratios: Vec<(i32, f64)> = test_f1s
        .iter()
        .copied()
        .zip(ratio_percents)
        .map(|(f1_score, ratio_percent)| (ratio_percent, f1_score))
        .collect();

    let confidence_intervals: Vec<f64> = test_f1s.iter().map(|_| std).collect();

    let rigde_f1_score = get_ridge_regression_f1_score(samples);

    plot_learning_curve_with_confidence_intervals(
        &f1_with_ratios,
        filename,
        "F1 score",
        &format!("{filename}.png"),
        &confidence_intervals,
        rigde_f1_score,
    )
    .unwrap();
}

fn learning_curve_test_svm(samples: &[Sample], filename: &str) {
    const KERNEL: KernelType = KernelType::RBF { gamma: 0.5 };
    const SVM_REGULARIZATION: f64 = 0.1;
    const TOLERANCE: f64 = 0.01;
    const MAX_ITERATIONS: usize = 2;

    let mut test_f1s = vec![];

    let ratio_percents = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90];

    for ratio_percent in ratio_percents {
        println!("calculating for ratio percent {ratio_percent}...");

        let ratio = ratio_percent as f64 / 100.0;
        let (train_samples, test_samples) = split(samples, ratio);

        let train_matrix = convert_samples_to_matrix(&train_samples);
        let train_labels = convert_labels_to_vector(&train_samples);
        let test_matrix = convert_samples_to_matrix(&test_samples);

        let mut svm_model =
            SupportVectorMachine::new(KERNEL, SVM_REGULARIZATION, TOLERANCE, MAX_ITERATIONS);
        svm_model.fit(&train_matrix, &train_labels);

        let mut test_predictions = Vec::with_capacity(test_samples.len());

        for i in 0..test_matrix.nrows() {
            let features = test_matrix.row(i).transpose();

            let prediction = svm_model.predict(&features);
            test_predictions.push(prediction);
        }

        let test_f1 = calculate_f1_score(&test_samples, &test_predictions);
        test_f1s.push(test_f1);
    }

    let std = get_std(&test_f1s);

    let f1_with_ratios: Vec<(i32, f64)> = test_f1s
        .iter()
        .copied()
        .zip(ratio_percents)
        .map(|(f1_score, ratio_percent)| (ratio_percent, f1_score))
        .collect();

    let confidence_intervals: Vec<f64> = test_f1s.iter().map(|_| std).collect();

    let rigde_f1_score = get_ridge_regression_f1_score(samples);

    plot_learning_curve_with_confidence_intervals(
        &f1_with_ratios,
        filename,
        "F1 score",
        &format!("{filename}.png"),
        &confidence_intervals,
        rigde_f1_score,
    )
    .unwrap();
}

fn get_ridge_regression_f1_score(samples: &[Sample]) -> f64 {
    const TRAIN_RATIO: f64 = 0.6;
    let (train_samples, test_samples) = split(samples, TRAIN_RATIO);

    pub const REGULARIZATION: f64 = 0.0005;
    let mut model = RidgeRegression::new(REGULARIZATION);
    model.fit(&train_samples);

    let mut test_predictions = Vec::with_capacity(test_samples.len());

    for sample in &test_samples {
        let prediction = model.predict(&sample.features);
        test_predictions.push(prediction);
    }

    calculate_f1_score(&test_samples, &test_predictions)
}
