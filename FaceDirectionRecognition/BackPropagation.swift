//
//  BackPropagation.swift
//  Handwritten Recognition
//
//  Created by Ashis Laha on 16/03/17.
//  Copyright © 2017 Ashis Laha. All rights reserved.
//

import Foundation

struct MathConstants {
    static let e : Double = 2.71828
}

struct LearningConstants {
    static let learning_rate = 0.2
    static let momentum      = 0.1
    static let initialWeightValue = 0.15
}

class BackPropagation {

/*
     n_inout  = Number of input units
     n_hidden = Number of units in hidden layer
     n_output = Number of units in output layer
     
     eta = learning rate
     momentum = considering previous output
     
     x_ji   = ith input to jth node
     w_ji   = weightage from ith node to jth node
     net_j  = Summation( x_ji * w_ji ) over all i
     o_j    = output at jth node
     t_j    = target output at jth node 
     sigmoid(x) = Sigmoid func of input x
     
     // ERROR in jth node at output layer 
     
     delta_j = o_j * (1 - o_j) * (t_j - o_j)
     
     // ERROR in jth node at Hidden layer 
     
     delta_j = o_j * (1 - o_j) * Summation( w_kj * delta_k ) where k belongs to outputs 
     
     // Update Each network wight w_ji 
     
     w_ji(n) = w_ji + eta * delta_j * x_ji + momentum * w_ji(n-1)
 
 */
    
    
    fileprivate var n_input  : Int = 0
    fileprivate var n_hidden : Int = 0
    fileprivate var n_output : Int = 0
    
    fileprivate var inputs         = [Double]() // Defined in the Training Examples
    fileprivate var target_outputs = [Double]() // Defined in the Training Examples
    
    fileprivate var weightages_from_input_to_hidden  = [[Double]]()
    fileprivate var outputs_hidden_layers            = [Double]()
    fileprivate var weightages_from_hidden_to_output = [[Double]]()
    fileprivate var actual_outputs = [Double]()
    
    fileprivate var errors_output_layer = [Double]()
    fileprivate var errors_hidden_layer = [Double]()
    
    /* 
        the output of back-propagation is weightages_from_input_to_hidden & wightages_from_hidden_to_output 2-D matrix
    */
    
    
    required init(units_input_layer : Int , units_hidden_layer : Int, units_output_layer : Int) {
        n_input  = units_input_layer
        n_hidden = units_hidden_layer
        n_output = units_output_layer
        initializeParameters()
    }
    
    fileprivate func initializeParameters() {
        defaultInitializeWeightVector()
        initializeError()
    }
    
    fileprivate func defaultInitializeWeightVector() {
        for _ in 0..<n_hidden {
            var temp = [Double]()
            for _ in 0..<n_input {
                temp.append(LearningConstants.initialWeightValue)
            }
            weightages_from_input_to_hidden.append(temp)
        }
        
        for _ in 0..<n_output {
            var temp = [Double]()
            for _ in 0..<n_hidden {
                temp.append(LearningConstants.initialWeightValue)
            }
            weightages_from_hidden_to_output.append(temp)
        }
    }
    
    fileprivate func initializeError() {
        for _ in 0..<n_hidden { outputs_hidden_layers.append(0.0) ; errors_hidden_layer.append(0.0) }
        for _ in 0..<n_output { actual_outputs.append(0.0); errors_output_layer.append(0.0) }
    }
    
    //MARK:- Assign Weight Vector till last Training happens 
    
    func intializeWeightVectorParametersTillLastTrain(initialWeightInputToHidden : [[Double]], initialWeightHiddenToOutput : [[Double]] ) {
        
        for hidden_index in 0..<n_hidden {
            for input_index in 0..<n_input {
                weightages_from_input_to_hidden[hidden_index][input_index] = initialWeightInputToHidden[hidden_index][input_index]
            }
        }
        
        for output_index in 0..<n_output {
            for hidden_index in 0..<n_hidden {
                weightages_from_hidden_to_output[output_index][hidden_index] = initialWeightHiddenToOutput[output_index][hidden_index]
            }
        }
    }
    
    //MARK:- Train ( single iteration )
    
    func train(inputs_to_ANN : [Double] , target_outputs_to_ANN : [Double] ) {
        inputs = inputs_to_ANN
        target_outputs = target_outputs_to_ANN
        
        // STEP 1 : Propagate the input forward through network and calculate Output
        
        // calculate sigmoid output at hidden layer units
        
        calculateOutputAtHiddenLayer()
        
        // calculate sigmoid actual outputs at Output layer units
        
        calculateOutputAtOutputLayer()
        
        // STEP 2 : Propagate the ERROR back-ward through network
        
        // calculate error at output layer
        
        calculateErrorAtOutputLayer()
        
        // calculate error at hidden layer 
        
        calculateErrorAtHiddenLayer()
        
        //STEP 3 : update the Weight vector
        
        // update weight from hidden to output
        
        updateWeightVectorFromHiddenToOutput()
        
        // update weight from input to hidden
        
        updateWeightVectorFromInputToHidden()
    }
    
    //MARK:- Calculate Output for unknown input
    
    func calculateOutput(inputs_to_ANN : [Double] ) -> [Double] {
        inputs = inputs_to_ANN
        calculateOutputAtHiddenLayer()
        calculateOutputAtOutputLayer()
        return actual_outputs
    }
    
    func getWeightVector() ->  [[[Double]]] {
        return [weightages_from_input_to_hidden,weightages_from_hidden_to_output]
    }

    
    //MARK:- Output calculation
    
    fileprivate func calculateOutputAtHiddenLayer() {
        for index_hidden_layer in 0..<n_hidden {
            var summation = 0.0
            for index_input_layer in 0..<n_input {
                summation = summation + inputs[index_input_layer] * weightages_from_input_to_hidden[index_hidden_layer][index_input_layer]
            }
            outputs_hidden_layers[index_hidden_layer] = sigmoid(input: summation)
        }
        
        // take the normalized value
        let total = outputs_hidden_layers.reduce(0.0, +)
        for hidden_index in 0..<outputs_hidden_layers.count {
            outputs_hidden_layers[hidden_index] /= total;
        }
        
    }
    
    fileprivate func calculateOutputAtOutputLayer() { // actual output at output layer
        for index_output_layer in 0..<n_output {
            var summation = 0.0
            for index_hidden_layer in 0..<n_hidden {
                summation = summation + outputs_hidden_layers[index_hidden_layer] * weightages_from_hidden_to_output[index_output_layer][index_hidden_layer]
            }
            actual_outputs[index_output_layer] = sigmoid(input: summation)
        }
        
        // take the normalized value
        let total = actual_outputs.reduce(0.0, +)
        for output_index in 0..<actual_outputs.count {
            actual_outputs[output_index] /= total;
        }
    }
    
    
    //MARK:- Error Calculation
    
    fileprivate func calculateErrorAtOutputLayer() {
        for index_output_layer in 0..<n_output {
            let actual = actual_outputs[index_output_layer]
            let target = target_outputs[index_output_layer]
            errors_output_layer[index_output_layer] = actual * (1-actual) * (target - actual)
        }
    }
    
    fileprivate func calculateErrorAtHiddenLayer() {
        for index_hidden_layer in 0..<n_hidden {
            let actual = outputs_hidden_layers[index_hidden_layer]
            var error_sum_from_output = 0.0
            for index_output_layer in 0..<n_output {
                error_sum_from_output = error_sum_from_output + errors_output_layer[index_output_layer] * weightages_from_hidden_to_output[index_output_layer][index_hidden_layer]
            }
            errors_hidden_layer[index_hidden_layer] = actual * (1-actual) * error_sum_from_output
        }
    }
    
    //MARK:- Update Weight vector
    
    fileprivate func updateWeightVectorFromHiddenToOutput() {
        for index_hidden_layer in 0..<n_hidden {
            for index_output_layer in 0..<n_output {
                let errorDelta = LearningConstants.learning_rate * errors_output_layer[index_output_layer] * outputs_hidden_layers[index_hidden_layer]
                let momentumDelta = LearningConstants.momentum * weightages_from_hidden_to_output[index_output_layer][index_hidden_layer]
                
                weightages_from_hidden_to_output[index_output_layer][index_hidden_layer] += errorDelta + momentumDelta
            }
        }
    }
    
    fileprivate func updateWeightVectorFromInputToHidden() {
        for index_input_layer in 0..<n_input {
            for index_hidden_layer in 0..<n_hidden {
                let errorDelta = LearningConstants.learning_rate * errors_hidden_layer[index_hidden_layer] * inputs[index_input_layer]
                let momentumDelta = LearningConstants.momentum * weightages_from_input_to_hidden[index_hidden_layer][index_input_layer]
                
                weightages_from_input_to_hidden[index_hidden_layer][index_input_layer] += errorDelta + momentumDelta
            }
        }
    }
    
    func saveWeightVector(weightages_from_input_to_hidden : [[Double]], weightages_from_hidden_to_output : [[Double]]) {
        self.weightages_from_input_to_hidden  = weightages_from_input_to_hidden
        self.weightages_from_hidden_to_output = weightages_from_hidden_to_output
    }
    
    
    //MARK:- Sigmoid Output 
    
    fileprivate func sigmoid(input : Double) -> Double {
        return 1.0 / ( 1.0 + pow(MathConstants.e, -1.0 * input))
    }
    
}
