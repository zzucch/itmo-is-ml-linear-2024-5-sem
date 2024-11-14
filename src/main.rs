use linear::linear_regression::Data;

fn csv_entries_to_data(entries: Vec<linear::parse::CsvEntry>) -> Vec<Data> {
    entries
        .into_iter()
        .map(|entry| Data {
            features: entry.values.try_into().unwrap(),
            label: match entry.diagnosis {
                linear::parse::Diagnosis::Malignant => 1.0,
                linear::parse::Diagnosis::Benign => -1.0,
            },
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const DATA_FILEPATH: &str = "data/breast-cancer.csv";

    let entries = linear::parse::parse(DATA_FILEPATH)?;
    assert!(!entries.is_empty());
    assert_eq!(
        entries.first().unwrap().values.len(),
        linear::linear_regression::DIMENSIONS
    );

    let data = csv_entries_to_data(entries);

    println!("{:?}", data);

    Ok(())
}
