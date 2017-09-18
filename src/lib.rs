#![feature(test)]

extern crate test;

#[derive(Debug)]
pub struct Perceptron {
    learning_rate: f64, // constant value to alter the weights by per training iteration
    epochs: isize, // Number of times to iterate while training
    weights: Vec<f64>,
    base_weight: f64
}


impl Perceptron {
    pub fn new(learning_rate: f64, epochs: isize) -> Perceptron {
        Perceptron {
            learning_rate: learning_rate,
            epochs: epochs,
            weights: vec!(),
            base_weight: 0.5
        }
    }

    /// Output should probably be between 0.0 and 1.0 (I have no clue)
    /// As a thought experiment, I will throw some values around
    fn net_input(&self, input_vector: &Vec<f64>) -> f64 {
        let prediction = input_vector
            .iter()
            .zip(self.weights.iter())
            .fold(self.base_weight, | activation: f64, x: (&f64, &f64)| {
            let (input_value, related_weight) = x;
            activation + (input_value * related_weight)
        });
        prediction
    }

    pub fn predict(&self, prediction_vector: &Vec<f64>) -> f64 {
        if self.net_input(prediction_vector) > 0.0 {
            return 1.0
        } else {
            return -1.0
        }
    }

    //TODO replace this with a From trait impl
    fn normalize_to_bool(value: f64) -> bool {
        if value > 0.0 {
            true
        } else {
            false
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
        for _ in 0..self.epochs {
            for t in training_input_set {
                let mut t = t.clone();
                let expected_answer = t.pop().unwrap();

                let prediction = self.predict(&t);

                let error: f64 = expected_answer - prediction;

                self.base_weight = self.base_weight + self.learning_rate * error;

                for i in 0..(t.len() ) {
                    self.weights[i] = self.weights[i] + self.learning_rate * error * t[i]
                }
            }
        }
    }

    /// Testing method that sets the weight attributes.
    fn set_weights(&mut self, base: f64, weights: Vec<f64>) {
        self.base_weight = base;
        self.weights = weights;
    }

}


mod tests {

    use super::*;
    use test::Bencher;

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