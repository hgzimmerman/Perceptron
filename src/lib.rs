#![feature(test)]

extern crate test;

#[derive(Debug)]
pub struct Perceptron {
    learning_rate: f64, // constant value to alter the weights by per training iteration
    epochs: usize, // Number of times to iterate while training.
    weights: Vec<f64>,
    base_weight: f64
}


impl Perceptron {

    /// Constructor
    pub fn new(learning_rate: f64, epochs: usize) -> Perceptron {
        Perceptron {
            learning_rate: learning_rate,
            epochs: epochs,
            weights: vec!(),
            base_weight: 1.0 // I don't know what an appropriate base weight should be, 0 is obviously wrong, because it could never move from there.
        }
    }

    /// Output should probably be between -1.0 and 1.0.
    fn net_input(&self, input_vector: &Vec<f64>) -> f64 {
        input_vector
            .iter()
            .zip(self.weights.iter())
            .fold(self.base_weight, | activation: f64, x: (&f64, &f64)| {
            let (input_value, related_weight) = x;
            activation + (input_value * related_weight) // accumulate the current activation value added to the product.
        })
    }

    /// Predicts the outcome based on the perceptron's weights.
    /// 1.0 indicates a true while -1.0 indicates a false.
    pub fn predict(&self, prediction_vector: &Vec<f64>) -> f64 {
        if self.net_input(prediction_vector) > 0.0 {
            return 1.0
        } else {
            return -1.0
        }
    }


    /// An acceptable vector for an AND gate would be 1 1 1, or 1 0 0, or 0 0 0, or 0 1 0.
    /// This will loop for the number of times specified in the perceptron's epoch.
    /// Training consists of predicting the output, calculating the error and then adding the error times the learning rate to the weight
    /// Iterating in this way will cause the error to decrease over time, to the point where the weights can be used to make useful predictions.
    pub fn train(&mut self, training_input_set: &[Vec<f64>]) {
        let sample = &training_input_set[0];
        self.weights = vec![0.0; sample.len() - 1]; // initialize the vector to 0.0 with the size of the input set

        // loop for number of epochs
        for _ in 0..self.epochs -1 {
            for training_row in training_input_set {
                let mut training_row = training_row.clone();
                let expected_answer = training_row.pop().unwrap();

                let prediction = self.predict(&training_row);

                let error: f64 = expected_answer - prediction;

                self.base_weight = self.base_weight + self.learning_rate * error;

                for i in 0..(training_row.len() ) {
                    self.weights[i] = self.weights[i] + self.learning_rate * error * training_row[i]
                }
            }
        }
    }

    /// Set the weight attributes manually.
    pub fn set_weights(&mut self, base: f64, weights: Vec<f64>) {
        self.base_weight = base;
        self.weights = weights;
    }
}



mod tests {
    use super::*;

    #[test]
    fn prediction_and() {
        let mut perceptron: Perceptron = Perceptron::new(0.0, 1);
        perceptron.set_weights(-1.0, vec!(0.6, 0.6));
        assert_eq!(1.0, perceptron.predict(&vec!(1.0, 1.0)));
        assert_eq!(-1.0, perceptron.predict(&vec!(0.0, 1.0)));
    }

    #[test]
    fn prediction_or() {
        let mut perceptron: Perceptron = Perceptron::new(0.0, 1);
        perceptron.set_weights(0.6, vec!(1.0, 1.0));
        assert_eq!(1.0, perceptron.predict(&vec!(1.0, 1.0)))
    }


    #[test]
    fn training_test() {
        let mut perceptron: Perceptron = Perceptron::new(0.1, 30);
        perceptron.train(&[vec!(0.0, 0.0, -1.0), vec!(1.0, 1.0, 1.0), vec!(1.0, 0.0, -1.0), vec!(0.0, 1.0, -1.0)]);
        println!("{:?}", perceptron);
    }

    #[test]
    fn train_and_predict_and() {
        let mut perceptron: Perceptron = Perceptron::new(0.1, 5);
        perceptron.train(&[vec!(-1.0, -1.0, -1.0), vec!(1.0, 1.0, 1.0), vec!(1.0, -1.0, -1.0), vec!(-1.0, 1.0, -1.0)]);

        println!("Predicting with perceptron: {:?}", perceptron);
        assert_eq!(-1.0, perceptron.predict(&vec!(1.0, -1.0)));
        assert_eq!(-1.0, perceptron.predict(&vec!(-1.0, 1.0)));
        assert_eq!(-1.0, perceptron.predict(&vec!(-1.0, -1.0)));
        assert_eq!(1.0, perceptron.predict(&vec!(1.0, 1.0)));
    }

    #[test]
    fn train_and_predict_nand() {
        let mut perceptron: Perceptron = Perceptron::new(0.1, 5);
        perceptron.train(&[vec!(-1.0, -1.0, 1.0), vec!(1.0, 1.0, -1.0), vec!(1.0, -1.0, 1.0), vec!(-1.0, 1.0, 1.0)]);

        println!("Predicting with perceptron: {:?}", perceptron);
        assert_eq!(1.0, perceptron.predict(&vec!(1.0, -1.0)));
        assert_eq!(1.0, perceptron.predict(&vec!(-1.0, 1.0)));
        assert_eq!(1.0, perceptron.predict(&vec!(-1.0, -1.0)));
        assert_eq!(-1.0, perceptron.predict(&vec!(1.0, 1.0)));
    }

    #[test]
    fn train_and_predict_or() {
        let mut perceptron: Perceptron = Perceptron::new(0.1, 5);
        perceptron.train(&vec!(vec!(-1.0, -1.0, -1.0), vec!(1.0, 1.0, 1.0), vec!(1.0, -1.0, 1.0), vec!(-1.0, 1.0, 1.0)));

        println!("Predicting with perceptron: {:?}", perceptron);
        assert_eq!(1.0, perceptron.predict(&vec!(1.0, -1.0)));
        assert_eq!(1.0, perceptron.predict(&vec!(-1.0, 1.0)));
        assert_eq!(-1.0, perceptron.predict(&vec!(-1.0, -1.0)));
        assert_eq!(1.0, perceptron.predict(&vec!(1.0, 1.0)));
    }

    #[test]
    fn train_and_predict_nor() {
        let mut perceptron: Perceptron = Perceptron::new(0.1, 5);
        perceptron.train(&[vec!(-1.0, -1.0, 1.0), vec!(1.0, 1.0, -1.0), vec!(1.0, -1.0, -1.0), vec!(-1.0, 1.0, -1.0)]);

        println!("Predicting with perceptron: {:?}", perceptron);
        assert_eq!(-1.0, perceptron.predict(&vec!(1.0, -1.0)));
        assert_eq!(-1.0, perceptron.predict(&vec!(-1.0, 1.0)));
        assert_eq!(1.0, perceptron.predict(&vec!(-1.0, -1.0)));
        assert_eq!(-1.0, perceptron.predict(&vec!(1.0, 1.0)));
    }

    #[test]
    fn train_and_predict_invert() {
        let mut perceptron: Perceptron = Perceptron::new(0.1, 5);
        perceptron.train(&[vec!(-1.0, 1.0), vec!(1.0, -1.0)]);

        println!("Predicting with perceptron: {:?}", perceptron);
        assert_eq!(-1.0, perceptron.predict(&vec!(1.0)));
        assert_eq!(1.0, perceptron.predict(&vec!(-1.0)));
    }

    #[test]
    fn train_and_predict_identity() {
        let mut perceptron: Perceptron = Perceptron::new(0.1, 5);
        perceptron.train(&[vec!(-1.0, -1.0), vec!(1.0, 1.0)]);

        println!("Predicting with perceptron: {:?}", perceptron);
        assert_eq!(1.0, perceptron.predict(&vec!(1.0)));
        assert_eq!(-1.0, perceptron.predict(&vec!(-1.0)));
    }

    mod benches {
        use test::Bencher;
        use super::*;

        #[bench]
        fn train_and_operation_1000_times(b: &mut Bencher) {
            let mut perceptron: Perceptron = Perceptron::new(0.1, 1000);
            b.iter(|| {
                perceptron.train(&[vec!(-1.0, -1.0, -1.0), vec!(1.0, 1.0, 1.0), vec!(1.0, -1.0, -1.0), vec!(-1.0, 1.0, -1.0)]);
            } );
        }

        #[bench]
        fn train_and_operation_1_times(b: &mut Bencher) {
            let mut perceptron: Perceptron = Perceptron::new(0.1, 1);
            b.iter(|| {
                perceptron.train(&[vec!(-1.0, -1.0, -1.0), vec!(1.0, 1.0, 1.0), vec!(1.0, -1.0, -1.0), vec!(-1.0, 1.0, -1.0)]);
            } );
        }
    }
}