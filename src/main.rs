extern crate csv;
extern crate rand;
#[macro_use]
extern crate serde_derive;


use std::error::Error;
use std::io;
use std::process;
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
                    for j in 0..$columns {
                        result[(i, j)] = rand::random();
                    }
                }

                result
            }

            #[allow(dead_code)]
            fn apply(&self, f: &Fn(f32) -> f32) -> Self {
                let mut result = $name::new();

                for i in 0..$rows {
                    for j in 0..$columns {
                        result[(i, j)] = f(self[(i, j)]);
                    }
                }

                result
            }

            fn element_mul(&self, other: &Self) -> Self {
                let mut result = Self::new();

                for i in 0..$rows {
                    for j in 0..$columns {
                        result[(i, j)] = self[(i, j)] * other[(i, j)];
                    }
                }

                result
            }

            fn sum(&self) -> f32 {
                let mut result = 0.0;

                for i in 0..$rows {
                    for j in 0..$columns {
                        result += self[(i, j)];
                    }
                }

                result
            }

            fn abs_sum(&self) -> f32 {
                let mut result = 0.0;

                for i in 0..$rows {
                    for j in 0..$columns {
                        result += self[(i, j)].abs();
                    }
                }

                result
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{} [\n", stringify!($name))?;
                for i in 0..$rows {
                    write!(f, "  [")?;
                    for j in 0..$columns {
                        if j < $columns - 1 {
                            write!(f, "{:?}, ", self[(i, j)])?
                        } else {
                            write!(f, "{:?}", self[(i, j)])?
                        }
                    }
                    write!(f, "]\n")?;
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

        impl IndexMut<(usize)> for $name {
            fn index_mut(&mut self, index: usize) -> &mut [f32; $columns] {
                &mut self.data[index]
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

        impl<'a> Add<&'a $name> for &'a $name {
            type Output = $name;

            fn add(self, other: &'a $name) -> $name {
                let mut result = $name::new();

                for i in 0..$rows {
                    for j in 0..$columns {
                        result[(i, j)] = self[(i, j)] + other[(i, j)];
                    }
                }

                result
            }
        }

        impl<'a> Mul<f32> for &'a $name {
            type Output = $name;

            fn mul(self, other: f32) -> $name {
                let mut result = $name::new();

                for i in 0..$rows {
                    for j in 0..$columns {
                        result[(i, j)] = self[(i, j)] * other;
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

macro_rules! gen_transpose {
    ($name:ident, $new_matrix_name:ident, $rows:expr, $columns:expr) => {
        impl $name {
            fn transpose(&self) -> $new_matrix_name {
                let mut result = $new_matrix_name::new();

                for i in 0..$rows {
                    for j in 0..$columns {
                        result[(i, j)] = self[(j, i)];
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
            use std::ops::{Index, IndexMut, Mul, Sub, Add};
            use std::f32::consts::E;

            matrix![Input, $num_examples; $input_nodes];

            matrix![W1, $input_nodes; $hidden_nodes];
            matrix![W1T, $hidden_nodes; $input_nodes];
            gen_transpose!(W1T, W1, $input_nodes, $hidden_nodes);

            matrix![Z2, $num_examples; $hidden_nodes];
            matrix![Z2T, $hidden_nodes; $num_examples];
            gen_transpose!(Z2, Z2T, $hidden_nodes, $num_examples);

            matrix![W2, $hidden_nodes; $output_nodes];
            matrix![W2T, $output_nodes; $hidden_nodes];
            gen_transpose!(W2, W2T, $output_nodes, $hidden_nodes);
            gen_transpose!(W2T, W2, $hidden_nodes, $output_nodes);

            matrix![Z3, $num_examples; $output_nodes];
            matrix![Z3T, $output_nodes; $num_examples];
            gen_transpose!(Z3, Z3T, $output_nodes, $num_examples);

            gen_mul!(Input, W1, Z2, $num_examples, $hidden_nodes);
            gen_mul!(Z2, W2, Z3, $num_examples, $output_nodes);
            gen_mul!(Z2T, Input, W1T, $hidden_nodes, $input_nodes);
            gen_mul!(Z3T, Z2, W2T, $output_nodes, $hidden_nodes);
            gen_mul!(Z3, W2T, Z2, $num_examples, $hidden_nodes);

            fn neg_ln(value: f32) -> f32 {
                (value.ln())
            }


            fn cross_entropy(guesses: &Z3, expected: &Z3, num_examples: f32) -> f32 {
                let left_sum = expected.apply(&|v| -v).element_mul(&guesses.apply(&neg_ln));
                let right_sum = expected.apply(&|v: f32| 1.0 - v).element_mul(&guesses.apply(&|v: f32| (1.0 - v).ln()));


                (1.0 / num_examples) * (&left_sum - &right_sum).sum()
            }

            fn regularization(w1: &W1, w2: &W2, lambda: f32, num_examples: f32) -> f32 {
                (lambda / ( 2.0 * num_examples)) * w1.sum() + w2.sum()
            }


            fn sigmoid(value: f32) -> f32 {
                1.0 / (1.0 + E.powf(-value))
            }

            fn sigmoid_prime(value: f32) -> f32 {
                let s = sigmoid(value);
                s * (1.0 - s)
            }

            fn squared_error(value: f32) -> f32 {
                0.5 * (value).abs().powf(2.0)
            }

            fn square(value: f32) -> f32 {
                value.powf(2.0)
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

                fn best_class(guess: &[f32]) -> usize {
                    guess.iter().enumerate().max_by(|(_, &v1), (_, &v2)| v1.partial_cmp(&v2).unwrap()).unwrap().0
                }

                pub fn accuracy(guesses: &Z3, expected: &Z3, num_examples: usize) -> f32 {
                    let mut correct_guesses = 0;

                    for i in 0..num_examples {
                        let guess_class = Self::best_class(&guesses[i]);
                        let expected_class = Self::best_class(&expected[i]);

                        if guess_class == expected_class {
                            correct_guesses += 1;
                        }
                    }

                    correct_guesses as f32 / num_examples as f32
                }

                pub fn train(&mut self, input: &Input, outputs: &Z3, learning_rate: f32, lambda: f32, num_iterations: usize) {
                    for i in 0..num_iterations {
                        let (_, w1_grad, w2_grad) = self.cost_with_gradient(input, outputs, lambda);
                        // println!("Iteration {}", i);
                        // println!("w1_grad = {:?}", w1_grad);
                        // println!("w2_grad = {:?}", w2_grad);
                        // println!("Cost = {:?}", cost);
                        // println!("");
                        self.w1 = &self.w1 - &(&w1_grad * learning_rate);
                        self.w2 = &self.w2 - &(&w2_grad * learning_rate);

                        let precision = w1_grad.abs_sum() + w2_grad.abs_sum();
                        if precision < 0.01 {
                            println!("Found desired precision after {} iterations", i + 1);
                            break;
                        }
                    }
                }

                pub fn predict(&self, input: &Input) -> Z3 {
                    let z2 = input * &self.w1;
                    let a2 = z2.apply(&sigmoid);
                    let z3 = &a2 * &self.w2;
                    let guesses = z3.apply(&sigmoid);

                    guesses
                }

                pub fn cost(&self, guesses: &Z3, expected: &Z3, lambda: f32) -> f32 {
                    let w1_copy = &self.w1.apply(&square);
                    let w2_copy = &self.w2.apply(&square);
                    let unregularized_cost = cross_entropy(guesses, expected, $num_examples as f32);
                    let regularized_cost = regularization(&w1_copy, &w2_copy, lambda, $num_examples as f32);

                    regularized_cost + unregularized_cost
                }

                fn cost_with_gradient(&mut self, input: &Input, outputs: &Z3, lambda: f32) -> (f32, W1, W2) {
                    // Forward propagate
                    let z2 = input * &self.w1;
                    let a2 = z2.apply(&sigmoid);
                    let z3 = &a2 * &self.w2;
                    let guesses = z3.apply(&sigmoid);

                    // Back propagate
                    let d3 = &guesses - &outputs;
                    let d2 = (&d3 * &self.w2.transpose()).element_mul(&z2.apply(&sigmoid_prime));
                    let delta_1 = &d2.transpose() * input;
                    let delta_2 = &d3.transpose() * &a2;

                    let num_examples = $num_examples as f32;
                    let w1_copy = &self.w1.apply(&square);
                    let w2_copy = &self.w2.apply(&square);
                    let w1 = &self.w1 * (lambda / num_examples);
                    let w2 = &self.w2 * (lambda / num_examples);

                    let w1_grad = &(&delta_1 * (1.0 / num_examples)).transpose() + &w1;
                    let w2_grad = &(&delta_2 * (1.0 / num_examples)).transpose() + &w2;

                    let unregularized_cost = cross_entropy(&guesses, outputs, num_examples);
                    let regularized_cost = regularization(&w1_copy, &w2_copy, lambda, num_examples);
                    let total_cost = unregularized_cost + regularized_cost;

                    (total_cost, w1_grad, w2_grad)
                }
            }
        }
    }
}


const NUM_EXAMPLES: usize = 75;
network!{
    name: my_network,
    input: {
        nodes: 4,
        examples: ::NUM_EXAMPLES,
    },
    layers: [{
        nodes: 10,
    }],
    output: {
        nodes: 3,
    }
}

#[derive(Debug,Deserialize)]
struct Example {
    sepal_length: f32,
    sepal_width: f32,
    petal_length: f32,
    petal_width: f32,
    class: String,
}


fn class(name: &str) -> [f32; 3] {
    match name {
        "Iris-setosa" => [1.0, 0.0, 0.0],
        "Iris-versicolor" => [0.0, 1.0, 0.0],
        "Iris-virginica" => [0.0, 0.0, 1.0],
        _ => panic!("Unknown class {}", name),
    }
}

fn read_data() -> Result<Vec<Example>, Box<Error>> {
    let mut builder = csv::ReaderBuilder::new();
    builder.has_headers(false);
    let mut rdr = builder.from_reader(io::stdin());
    let mut result = Vec::new();

    for entry in rdr.deserialize() {
        // The iterator yields Result<StringRecord, Error>, so we check the
        // error here..
        let record: Example = entry?;
        result.push(record);
    }


    Ok(result)
}

fn normalize_by<T, F>(values: &mut [T], get_value: F) where F: FnMut(&mut T) -> &mut f32, {
    let max = {
        get_value(values.iter_mut().max_by(|ref mut a, ref mut b| get_value(a).partial_cmp(&get_value(b)).unwrap()).unwrap())
    };
    let min = {
        get_value(values.iter_mut().min_by(|ref mut a, ref mut b| get_value(a).partial_cmp(&get_value(b)).unwrap()).unwrap())
    };

    values.iter_mut().for_each(|a| {
        let value = get_value(a);
        *value = (*value - *min) / (*max - *min);
    });
}


fn main() -> Result<(), Box<Error>> {
    use rand::{thread_rng, Rng};
    const LAMBDA: f32 = 1.0;
    const LEARNING_RATE: f32 = 0.33;


    let mut network = my_network::Network::new();
    let mut input = my_network::Input::new();
    let mut test_input = my_network::Input::new();
    let mut expected_output = my_network::Z3::new();
    let mut expected_test_output = my_network::Z3::new();
    let mut raw_data = read_data()?;
    thread_rng().shuffle(&mut raw_data);
    normalize_by(&mut raw_data, |data| &mut data.sepal_length);


    for i in 0..NUM_EXAMPLES {
        input[(i, 0)] = raw_data[i].sepal_length;
        input[(i, 1)] = raw_data[i].sepal_width;
        input[(i, 2)] = raw_data[i].petal_length;
        input[(i, 3)] = raw_data[i].petal_width;
        expected_output[i] = class(&raw_data[i].class);
    }

    for i in NUM_EXAMPLES..NUM_EXAMPLES * 2 {
        let normalized_index = i - NUM_EXAMPLES;
        test_input[(normalized_index, 0)] = raw_data[i].sepal_length;
        test_input[(normalized_index, 1)] = raw_data[i].sepal_width;
        test_input[(normalized_index, 2)] = raw_data[i].petal_length;
        test_input[(normalized_index, 3)] = raw_data[i].petal_width;
        expected_test_output[normalized_index] = class(&raw_data[i].class);
    }

    // Before training
    let untrained_guesses = network.predict(&input);
    let untrained_test_guesses = network.predict(&test_input);
    let untrained_cost = network.cost(&untrained_guesses, &expected_output, LAMBDA);
    let untrained_test_cost = network.cost(&untrained_test_guesses, &expected_test_output, LAMBDA);
    let untrained_accuracy = my_network::Network::accuracy(&untrained_guesses, &expected_output, NUM_EXAMPLES);
    let untrained_test_accuracy = my_network::Network::accuracy(&untrained_test_guesses, &expected_test_output, NUM_EXAMPLES);
    println!("Untrained cost: {}", untrained_cost);
    println!("Untrained test cost: {}", untrained_test_cost);
    println!("Untrained accuracy: {}%", untrained_accuracy * 100.0);
    println!("Untrained test accuracy: {}%", untrained_test_accuracy * 100.0);

    // Training
    println!("Training network...\n");
    network.train(&input, &expected_output, LEARNING_RATE, LAMBDA, 5000);

    // After training
    let trained_guesses = network.predict(&input);
    let trained_test_guesses = network.predict(&test_input);
    let trained_cost = network.cost(&trained_guesses, &expected_output, LAMBDA);
    let trained_test_cost = network.cost(&trained_test_guesses, &expected_test_output, LAMBDA);
    let trained_accuracy = my_network::Network::accuracy(&trained_guesses, &expected_output, NUM_EXAMPLES);
    let trained_test_accuracy = my_network::Network::accuracy(&trained_guesses, &expected_test_output, NUM_EXAMPLES);
    println!("Trained cost: {}", trained_cost);
    println!("Trained test cost: {}", trained_test_cost);
    println!("Trained accuracy: {}%", trained_accuracy * 100.0);
    println!("Trained test accuracy: {}%", trained_test_accuracy * 100.0);


    Ok(())
}
