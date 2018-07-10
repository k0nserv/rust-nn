#[macro_export]
macro_rules! matrix {
    ($name: ident, $rows:expr, $columns:expr) => {
        #[derive(Debug)]
        struct $name {
            data: [[f32; $columns], $rows],
        };
    }
}
