#![feature(test)]

extern crate test;
use std::convert::{From,Into};
use std::boxed::Box;

#[derive(Debug)]
pub struct Perceptron {
    learning_rate: f64, // constant value to alter the weights by per training iteration
    epochs: usize, // Number of times to iterate while training.
    weights: Vec<f64>, // The other values used in predictions, they are multiplied with the input values when calculating activation.
    base_weight: f64, // The first value used in predictions, not multiplied with input values when calculating activation. Sometimes referred to as a bias.
    output_connections: Vec<Connection>
}

#[derive(Debug)]
struct Connection {
    weight: f64,
    target: Box<Perceptron>
}

#[derive(Debug)]
struct NeuralNetwork {
    layers: Vec<Perceptron>
}



impl Perceptron {
    /// Constructor
    pub fn new(learning_rate: f64, epochs: usize) -> Perceptron {
        Perceptron {
            learning_rate: learning_rate,
            epochs: epochs,
            weights: vec!(),
            base_weight: 0.0,
            output_connections: vec!()
        }
    }

    /// Output should probably be between -1.0 and 1.0.
    fn net_input<T: Into<f64> + From<f64> + Clone>(&self, input_vector: &Vec<T>) -> f64 {
        input_vector
            .iter()
            .map(|x: &T| {
                x.clone().into()
            })
            .zip(self.weights.iter())
            .fold(self.base_weight, | activation: f64, x: (f64, &f64)| {
                let (input_value, related_weight): (f64, &f64) = x;
                activation + (input_value * related_weight) // accumulate the current activation value added to the product.
            })
    }

    /// Predicts the outcome based on the perceptron's weights.
    /// 1.0 indicates a true while -1.0 indicates a false.
    pub fn predict_raw(&self, prediction_vector: &Vec<f64>) -> f64 {
        if self.net_input(prediction_vector) > 0.0 {
            return 1.0
        } else {
            return -1.0
        }
    }

    fn predict<T: Into<f64> + From<f64> + Clone>(&self, prediction_vector: &Vec<T>) -> f64 {
        let prediction: T = T::from(self.net_input(prediction_vector));
        prediction.into()
    }


    /// An acceptable vector for an AND gate would be 1 1 1, or 1 0 0, or 0 0 0, or 0 1 0.
    /// This will loop for the number of times specified in the perceptron's epoch.
    /// Training consists of predicting the output, calculating the error and then adding the error times the learning rate to the weight
    /// Iterating in this way will cause the error to decrease over time, to the point where the weights can be used to make useful predictions.
    pub fn train<T: Into<f64> + From<f64> + Clone>(&mut self, training_input_set: &[Vec<T>]) {
        let sample = &training_input_set[0];
        self.weights = vec![0.0; sample.len() - 1]; // initialize the vector to 0.0 with the size of the input set

        // loop for number of epochs
        for _ in 0..self.epochs -1 {
            for training_row in training_input_set {
                let mut training_row = training_row.clone();
                let expected_answer: f64 = training_row.pop().unwrap().into();

                let prediction: f64 = self.predict(&training_row);

                let error: f64 = expected_answer - prediction;

                self.base_weight = self.base_weight + self.learning_rate * error;

                for i in 0..(training_row.len() ) {
                    let row: f64 = training_row[i].clone().into();
                    self.weights[i] = self.weights[i] + self.learning_rate * error * row;
                }
            }
        }
    }

    /// Set the weight attributes manually.
    pub fn set_weights(&mut self, base: f64, weights: Vec<f64>) {
        self.base_weight = base;
        self.weights = weights;
    }

    fn add_output_connection(&mut self, output_connection: Connection) {
        self.output_connections.push(output_connection);
    }
}



#[derive(Clone, Debug, PartialEq)]
enum Boolean {
    True,
    False
}

impl Into<f64> for Boolean {
    fn into(self) -> f64 {
        match self {
            Boolean::True => 1.0,
            Boolean::False => -1.0
        }
    }
}

impl From<f64> for Boolean {
    fn from(value: f64) -> Boolean {
        if value > 0.0 {
            Boolean::True
        } else {
            Boolean::False
        }
    }
}


mod tests {
    use super::*;

    #[test]
    fn prediction_and() {
        let mut perceptron: Perceptron = Perceptron::new(0.0, 1);
        perceptron.set_weights(-1.0, vec!(0.6, 0.6));
        assert_eq!(1.0, perceptron.predict(&vec!(True, True)));
        assert_eq!(-1.0, perceptron.predict(&vec!(False, True)));
    }

    #[test]
    fn prediction_or() {
        let mut perceptron: Perceptron = Perceptron::new(0.0, 1);
        perceptron.set_weights(0.6, vec!(1.0, 1.0));
        assert_eq!(True, perceptron.predict(&vec!(True, True)).into())
    }


    #[test]
    fn training_test() {
        let mut perceptron: Perceptron = Perceptron::new(0.1, 30);
        perceptron.train(&[vec!(0.0, 0.0, -1.0), vec!(1.0, 1.0, 1.0), vec!(1.0, 0.0, -1.0), vec!(0.0, 1.0, -1.0)]);
        println!("{:?}", perceptron);
    }


    use Boolean::*;
    #[test]
    fn train_and_predict_and() {
        let mut perceptron: Perceptron = Perceptron::new(0.1, 5);
        perceptron.train(&[vec!(False, False, False), vec!(True, True, True), vec!(True, False, False), vec!(False, True, False)]);

        println!("Predicting with perceptron: {:?}", perceptron);
        assert_eq!(False, perceptron.predict(&vec!(True, False)).into());
        assert_eq!(False, perceptron.predict(&vec!(False, True)).into());
        assert_eq!(False, perceptron.predict(&vec!(False, False)).into());
        assert_eq!(True, perceptron.predict(&vec!(True, True)).into());
    }

    #[test]
    fn train_and_predict_nand() {
        let mut perceptron: Perceptron = Perceptron::new(0.1, 5);
        perceptron.train(&[vec!(False, False, True), vec!(True, True, False), vec!(True, False, True), vec!(False, True, True)]);

        println!("Predicting with perceptron: {:?}", perceptron);
        assert_eq!(1.0, perceptron.predict(&vec!(True, False)));
        assert_eq!(1.0, perceptron.predict(&vec!(False, True)));
        assert_eq!(1.0, perceptron.predict(&vec!(False, False)));
        assert_eq!(-1.0, perceptron.predict(&vec!(True, True)));
    }

    #[test]
    fn train_and_predict_or() {
        let mut perceptron: Perceptron = Perceptron::new(0.1, 5);
        perceptron.train(&vec!(vec!(False, False, False), vec!(True, True, True), vec!(True, False, True), vec!(False, True, True)));

        println!("Predicting with perceptron: {:?}", perceptron);
        assert_eq!(1.0, perceptron.predict(&vec!(True, False)));
        assert_eq!(1.0, perceptron.predict(&vec!(False, True)));
        assert_eq!(-1.0, perceptron.predict(&vec!(False, False)));
        assert_eq!(1.0, perceptron.predict(&vec!(True, True)));
    }

    #[test]
    fn train_and_predict_nor() {
        let mut perceptron: Perceptron = Perceptron::new(0.1, 5);
        perceptron.train(&[vec!(False, False, True), vec!(True, True, False), vec!(True, False, False), vec!(False, True, False)]);

        println!("Predicting with perceptron: {:?}", perceptron);
        assert_eq!(-1.0, perceptron.predict(&vec!(True, False)));
        assert_eq!(-1.0, perceptron.predict(&vec!(False, True)));
        assert_eq!(1.0, perceptron.predict(&vec!(False, False)));
        assert_eq!(-1.0, perceptron.predict(&vec!(True, True)));
    }

//    #[test]
//    fn train_and_predict_xor() {
//
//        let mut and_perceptron: Perceptron = Perceptron::new(0.1, 5);
//        and_perceptron.train(&[vec!(False, False, False), vec!(True, True, True), vec!(True, False, False), vec!(False, True, False)]);
//        let mut or_perceptron: Perceptron = Perceptron::new(0.1, 5);
//        or_perceptron.train(&[vec!(False, False, False), vec!(True, True, True), vec!(True, False, True), vec!(False, True, True)]);
//        let mut xor_perceptron: Perceptron = Perceptron::new(0.1, 5);
//        xor_perceptron.train(&[vec!(False, False, True), vec!(True, True, False), vec!(True, False, False), vec!(False, True, False)]);
//
//        println!("Predicting with perceptron: {:?}", xor_perceptron);
//        assert_eq!(-1.0, xor_perceptron.predict(&vec!(True, False)));
//        assert_eq!(-1.0, xor_perceptron.predict(&vec!(False, True)));
//        assert_eq!(1.0, xor_perceptron.predict(&vec!(False, False)));
//        assert_eq!(-1.0, xor_perceptron.predict(&vec!(True, True)));
//    }

    #[test]
    fn train_and_predict_invert() {
        let mut perceptron: Perceptron = Perceptron::new(0.1, 5);
        perceptron.train(&[vec!(False, True), vec!(True, False)]);

        println!("Predicting with perceptron: {:?}", perceptron);
        assert_eq!(-1.0, perceptron.predict(&vec!(True)));
        assert_eq!(1.0, perceptron.predict(&vec!(False)));
    }

    #[test]
    fn train_and_predict_identity() {
        let mut perceptron: Perceptron = Perceptron::new(0.1, 5);
        perceptron.train(&[vec!(False, False), vec!(True, True)]);

        println!("Predicting with perceptron: {:?}", perceptron);
        assert_eq!(True, perceptron.predict(&vec!(True)).into());
        assert_eq!(False, perceptron.predict(&vec!(False)).into());
    }

    mod benches {
        use test::Bencher;
        use super::*;

        #[bench]
        fn train_and_operation_1000_times(b: &mut Bencher) {
            let mut perceptron: Perceptron = Perceptron::new(0.1, 1000);
            b.iter(|| {
                perceptron.train(&[vec!(False, False, False), vec!(True, True, True), vec!(True, False, False), vec!(False, True, False)]);
            } );
        }

        #[bench]
        fn train_and_operation_1_times(b: &mut Bencher) {
            let mut perceptron: Perceptron = Perceptron::new(0.1, 1);
            b.iter(|| {
                perceptron.train(&[vec!(False, False, False), vec!(True, True, True), vec!(True, False, False), vec!(False, True, False)]);
            } );
        }
    }
}