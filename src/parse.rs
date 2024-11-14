use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

#[derive(Debug)]
pub struct CsvEntry {
    pub diagnosis: Diagnosis,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Diagnosis {
    Malignant,
    Benign,
}

pub fn to_diagnosis(diagnosis: &str) -> Diagnosis {
    match diagnosis {
        "M" => Diagnosis::Malignant,
        "B" => Diagnosis::Benign,
        val => panic!("unexpected diagnosis {val}"),
    }
}

pub fn opposite_diagnosis(target: Diagnosis) -> Diagnosis {
    match target {
        Diagnosis::Malignant => Diagnosis::Benign,
        Diagnosis::Benign => Diagnosis::Malignant,
    }
}

pub fn z_score_normalize(data: &[f64]) -> Vec<f64> {
    let mean = data.iter().copied().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let std_dev = variance.sqrt();

    data.iter().map(|&x| (x - mean) / std_dev).collect()
}

pub fn parse(file_path: &str) -> Result<Vec<CsvEntry>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(BufReader::new(file));

    let mut entries = Vec::new();
    let mut values_list = Vec::new();

    for result in reader.records() {
        const DIAGNOSIS_FIELD_INDEX: usize = 1;

        let record = result?;

        let diagnosis_str = record.get(DIAGNOSIS_FIELD_INDEX).unwrap();
        let diagnosis = to_diagnosis(diagnosis_str);

        let values: Vec<f64> = record
            .iter()
            .enumerate()
            .filter_map(|(index, value)| {
                if index <= DIAGNOSIS_FIELD_INDEX {
                    None
                } else {
                    value.parse::<f64>().ok()
                }
            })
            .collect();

        values_list.push(values.clone());

        entries.push(CsvEntry { diagnosis, values });
    }

    let normalized_values = z_score_normalize(&values_list.concat());

    let value_length = entries.first().map_or(0, |entry| entry.values.len());

    for (entry, new_values) in entries
        .iter_mut()
        .zip(normalized_values.chunks(value_length))
    {
        entry.values = new_values.to_vec();
    }

    Ok(entries)
}
