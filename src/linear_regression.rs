pub const DIMENSIONS: usize = 30;

#[derive(Debug, Clone, Copy)]
pub struct Data {
    pub features: [f64; DIMENSIONS],
    pub label: f64, // 1 if malignant, -1 if benign
}
