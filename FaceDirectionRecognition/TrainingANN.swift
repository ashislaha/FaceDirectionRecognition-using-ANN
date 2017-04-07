//
//  TrainingANN.swift
//  Handwritten Recognition
//
//  Created by Ashis Laha on 17/03/17.
//  Copyright © 2017 Ashis Laha. All rights reserved.
//

import Foundation
import CoreData
import UIKit

// Define Artificial Neural Network constants 

struct ANNConstants {
    static let numberOfUnitsAtInputLayer  = ImageProcessingConstants.optimalDimensionX*ImageProcessingConstants.optimalDimensionY
    static let numberOfUnitsAtHiddenLayer = 3     // Experimental value
    static let numberOfUnitsAtOutputLayer = 4     // { left, right, straight, up }
}

struct FaceRecognitionDirectionConstants {
    static let left     = [0.9,0.1,0.1,0.1]
    static let right    = [0.1,0.1,0.1,0.9]
    static let top      = [0.1,0.9,0.1,0.1]
    static let straight = [0.1,0.1,0.9,0.1]
}

class TrainingANN {

    static let shared = TrainingANN()
    var backPropagation : BackPropagation?
    
    // use for saving in core data
    
    var faceDirection = [Double]()
    var inputsTrainData = [Double]()
    var pixels = [[PixelData]]()
    
    //MARK:- Train Back Propagation
    
    func trainBackpropagation() {
        backPropagation = BackPropagation(units_input_layer: ANNConstants.numberOfUnitsAtInputLayer, units_hidden_layer: ANNConstants.numberOfUnitsAtHiddenLayer, units_output_layer: ANNConstants.numberOfUnitsAtOutputLayer)
        initializeWeightVector()
        retrieveData()
        saveWeights()
    }
    
    //MARK:- Output for unknown input
    
    func getOutput(inputToANN : [Double]) -> [Double] {
        return backPropagation?.calculateOutput(inputs_to_ANN: inputToANN) ?? []
    }
    
    //MARK:- Weight Vector 
    
    func getWeightVector() -> [[[Double]]] {
        return backPropagation?.getWeightVector() ?? []
    }
    
    func initializeWeightVector() {
        let intialWeightVector = retrieveWeightVector()
        if intialWeightVector.count > 0 {
            let inputToHiddenWeight = intialWeightVector[0]
            let hiddenToOutput      = intialWeightVector[1]
            if inputToHiddenWeight.count > 0 && hiddenToOutput.count > 0 {
                backPropagation?.intializeWeightVectorParametersTillLastTrain(initialWeightInputToHidden: inputToHiddenWeight, initialWeightHiddenToOutput: hiddenToOutput)
            }
        }
    }
    
    //MARK:- Retrieve from Core Data 
    
    fileprivate func retrieveData() {
        
        guard let appDelegate = UIApplication.shared.delegate as? AppDelegate else { return }
        let managedObjectContext = appDelegate.persistentContainer.viewContext
        let fetchRequest = NSFetchRequest<NSFetchRequestResult>(entityName: "FaceImage")
        
        var faceImages = [FaceImage]()
        do {
            if let results = try managedObjectContext.fetch(fetchRequest) as? [FaceImage] {
                faceImages = results
            }
        } catch let error as NSError {
            print("Could not fetch : \(error)")
        }
        // give the faceImages to ANN 
        
        for faceImage in faceImages {
            if let inputs = faceImage.trainData, let direction = faceImage.direction {
                backPropagation?.train(inputs_to_ANN: inputs, target_outputs_to_ANN: direction)
            }
        }
    }
    
    
    //MARK:- Save weightVector in a JSON file into Document Directory
    
    fileprivate func saveWeights() {
        do {
            guard let directory = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true).first else { return }
            guard let pathUrl = NSURL(string: "file://\(directory)") else { return }
            guard let path = pathUrl.appendingPathComponent("WeightVector.json") else { return }
            
            guard let data = try? JSONSerialization.data(withJSONObject: backPropagation?.getWeightVector() ?? [], options: .prettyPrinted) else { return }
            try data.write(to: path)
        } catch let error {
            print(error.localizedDescription)
        }
    }
    
    fileprivate func retrieveWeightVector() -> [[[Double]]] {
        
        guard let directory = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true).first else { return [] }
        guard let pathUrl = NSURL(string: "file://\(directory)") else { return [] }
        guard let path = pathUrl.appendingPathComponent("WeightVector.json") else { return [] }
        
        guard let data = try? Data(contentsOf: path) else { return [] }
        guard let json = try? JSONSerialization.jsonObject(with: data, options: .allowFragments) else { return [] }
        if let doubleArray = json as? [[[Double]]] {
            return doubleArray
        }
        return []
    }
}
