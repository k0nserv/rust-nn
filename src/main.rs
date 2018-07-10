use std::fmt;

// #[macro_use]
// extern crate rust_nn;

#[macro_export]
macro_rules! matrix {
    [$name: ident, $rows:expr; $columns:expr] => {
        pub struct $name {
            data: [[f32; $columns]; $rows],
        }

        impl $name {
            pub fn new() -> Self {
                Self {
                    data: [[0.0; $columns]; $rows],
                }
            }

            #[allow(dead_code)]
            fn new_random() -> Self {
                let mut result = Self::new();

                for i in 0..$rows {
                    result.data[i] = rand::random();
                }

                result
            }

            #[allow(dead_code)]
            fn apply(mut self, f: &Fn(f32) -> f32) -> Self {
                for i in 0..$rows {
                    for j in 0..$columns {
                        self[(i, j)] = f(self[(i, j)]);
                    }
                }

                self
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{} [\n", stringify!($name))?;
                for i in 0..$rows {
                    write!(f, "  {:?},\n", self[i])?;
                }

                return write!(f, "]");
            }
        }


        impl Index<usize> for $name {
            type Output = [f32; $columns];

            fn index(&self, index: usize) -> &[f32; $columns] {
                &self.data[index]
            }
        }

        impl Index<(usize, usize)> for $name {
            type Output = f32;

            fn index(&self, index: (usize, usize)) -> &f32 {
                &self.data[index.0][index.1]
            }
        }

        impl IndexMut<(usize, usize)> for $name {
            fn index_mut(&mut self, index: (usize, usize)) -> &mut f32 {
                &mut self.data[index.0][index.1]
            }
        }

        impl<'a> Sub<&'a $name> for &'a $name {
            type Output = $name;

            fn sub(self, other: &'a $name) -> $name {
                let mut result = $name::new();

                for i in 0..$rows {
                    for j in 0..$columns {
                        result[(i, j)] = self[(i, j)] - other[(i, j)];
                    }
                }

                result
            }
        }
    }
}

macro_rules! gen_mul {
    ($lhs_name:ident, $rhs_name:ident, $output_name:ident, $lhs_rows:expr, $rhs_columns:expr) => {
        impl<'a> Mul<&'a $rhs_name> for &'a $lhs_name {
            type Output = $output_name;

            fn mul(self, other: &'a $rhs_name) -> $output_name {
                let mut result = $output_name::new();

                for i in 0..$lhs_rows {
                    for j in 0..$rhs_columns {
                        result[(i, j)] = self.data[i]
                            .iter()
                            .enumerate()
                            .fold(0.0, |acc, (idx, lhs)| acc + (lhs * other[(idx, j)]));
                    }
                }

                result
            }
        }
    };
}

#[macro_export]
macro_rules! network {
    {
        name: $name: ident,
        input: {
            nodes: $input_nodes: expr,
            examples: $num_examples: expr,
        },
        layers: [{
            nodes: $hidden_nodes: expr,
        }],
        output: {
            nodes: $output_nodes: expr,
        }
    } => {
        mod $name {
            extern crate rand;
            use std::fmt;
            use std::ops::{Index, IndexMut, Mul, Sub};
            use std::f32::consts::E;

            matrix![Input, $num_examples; $input_nodes];
            matrix![W1, $num_examples; $hidden_nodes];
            matrix![Z2, $num_examples; $hidden_nodes];
            matrix![W2, $num_examples; $output_nodes];
            gen_mul!(Input, W1, Z2, $num_examples, $hidden_nodes);
            matrix![Z3, $num_examples; $output_nodes];
            gen_mul!(Z2, W2, Z3, $num_examples, $output_nodes);

            fn sigmoid(value: f32) -> f32 {
                1.0 / (1.0 + E.powf(-value))
            }

            fn sigmoid_prime(value: f32) -> f32 {
                E.powf(-value) / ((1.0 + E.powf(-value)).powf(2.0))
            }

            fn squared_error(value: f32) -> f32 {
                0.5 * (value).abs().powf(2.0)
            }

            pub struct Network {
                w1: W1,
                w2: W2,
            }

            impl Network {
                pub fn new() -> Self {
                    Self {
                        w1: W1::new_random(),
                        w2: W2::new_random(),
                    }
                }

                pub fn forward_propagate(&self, input: &Input) -> Z3 {
                    let a2 = (input * &self.w1).apply(&sigmoid);
                    (&a2 * &self.w2).apply(&sigmoid)
                }

                pub fn train(&mut self, input: &Input, outputs: &Z3) {
                    let guesses = self.forward_propagate(input);
                    let error = (outputs - &guesses).apply(&squared_error);
                }
            }
        }
    }
}

network!{
    name: my_network,
    input: {
        nodes: 2,
        examples: 10,
    },
    layers: [{
        nodes: 3,
    }],
    output: {
        nodes: 1,
    }
}

fn main() {
    let network = my_network::Network::new();
    let input = my_network::Input::new();
    println!("{:?}", network.forward_propagate(&input));

    // // Input
    // matrix![InputMatrix, NUM_EXAMPLES; INPUT_NODES];
    // let mut input = InputMatrix::new();

    // input[(0, 0)] = 3.0;
    // input[(0, 1)] = 5.0;

    // input[(1, 0)] = 8.0;
    // input[(1, 1)] = 2.0;

    // // Hidden layer
    // matrix![W1, INPUT_NODES; HIDDEN_NODES];
    // let w1 = W1::new_random();
    // matrix![Z2, NUM_EXAMPLES; HIDDEN_NODES];
    // gen_mul!(InputMatrix, W1, Z2, NUM_EXAMPLES, HIDDEN_NODES);

    // // Output
    // matrix![W2, HIDDEN_NODES; OUTPUT_NODES];
    // let w2 = W2::new_random();
    // matrix![Z3, NUM_EXAMPLES; OUTPUT_NODES];
    // gen_mul!(Z2, W2, Z3, NUM_EXAMPLES, OUTPUT_NODES);

    // let a2 = (&input * &w1).apply(&sigmoid);
    // let y = (&a2 * &w2).apply(&sigmoid);
    // println!("{:?}", y);
}
