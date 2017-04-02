//
//  ViewController.swift
//  Handwritten Recognition
//
//  Created by Ashis Laha on 07/03/17.
//  Copyright © 2017 Ashis Laha. All rights reserved.
//

import UIKit
import CoreData

/* 
    "The FaceDirection Recognition" - Options <"left", "right", "top", "straight">
*/

enum FaceRecognitionDirection {
    case left       // (0.9,0.1,0.1,0.1)
    case right      // (0.1,0.1,0.1,0.9)
    case top        // (0.1,0.9,0.1,0.1)
    case straight   // (0.1,0.1,0.9,0.1)
    // because sigmoid func never generate 0/1 as an output, 0 < Output < 1
}

enum RecognitionType {
    case toTrainANN
    case recognizeUnknownData
}

class GetImageViewController: UIViewController {
    
    //MARK:- Outlets

    @IBOutlet weak var captureImageView: UIImageView! { didSet { captureImageView.isHidden = true }}
    @IBOutlet weak var recognizeTextBtnOutlet: UIButton! { didSet { recognizeTextBtnOutlet.isHidden = true }}
    @IBOutlet weak var textView: UITextView! { didSet{ textView.text = "" ; textView.isHidden = true ; textView.isUserInteractionEnabled = false }}
    @IBOutlet weak var optionsBtnOutlet: UIButton! { didSet { optionsBtnOutlet.isHidden = false } }
    @IBOutlet weak var normalizeIntoLowerDimension: UIButton! { didSet{ normalizeIntoLowerDimension.isHidden = true }}
    @IBOutlet weak var saveToTrainANN: UIButton! { didSet {saveToTrainANN.isHidden = true }}
    @IBOutlet weak var faceDirectionOutlet: UIButton! { didSet { faceDirectionOutlet.isHidden = true } }
    @IBOutlet weak var trainANNOutlet: UIButton! { didSet { trainANNOutlet.isHidden = true } }
    
    //MARK:- Variables
    
    var recognitionType : RecognitionType? = .toTrainANN // change while recognize
    var activityIndicator : UIActivityIndicatorView?
    
    //MARK:- View controller life cycle
    
    override func viewDidLoad() {
        super.viewDidLoad()
        //TrainingANN.shared.backPropagation?.retrieveWeightVector()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    //MARK:- Take Image from Camera or Photo library 
    
    @IBAction func takeImageBtn(_ sender: UIButton) {
        hideElements()
        getImage()
    }
    
    //MARK:- GetImage
    
    fileprivate func getImage() {
        
        self.captureImageView.image = nil
        // Show an action sheet with options 1. camera 2. Photo Library 3. Cancel
        let alertController = UIAlertController(title: "Get Image", message: "Take Image to recognize", preferredStyle: .actionSheet)
        let cameraAction = UIAlertAction(title: "Camera", style: .default) { [weak self] (action) in
            let imagePickerController = UIImagePickerController()
            imagePickerController.delegate = self
            imagePickerController.sourceType = .camera
            self?.present(imagePickerController, animated: true, completion: nil)
        }
        
        let photoLibraryAction = UIAlertAction(title: "Photo-Library", style: .default) { [weak self] (action) in
            let imagePickerController = UIImagePickerController()
            imagePickerController.delegate = self
            imagePickerController.sourceType = .photoLibrary
            self?.present(imagePickerController, animated: true, completion: nil)
        }
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        
        alertController.addAction(cameraAction)
        alertController.addAction(photoLibraryAction)
        alertController.addAction(cancelAction)
        self.present(alertController, animated: true, completion: nil)
    }

    
    //MARK:- Recognition Type
    
    @IBAction func options(_ sender: UIButton) {
        
        let alertController = UIAlertController(title: "Options for ADMIN ", message: "Options", preferredStyle: .actionSheet)
        let printedTextAction = UIAlertAction(title: "To Train ANN", style: .default) { [weak self] (action) in
            self?.recognitionType = .toTrainANN
            self?.hideElements()
        }
        
        let handWrittenAction = UIAlertAction(title: "Recognize the Unknown Data", style: .default) { [weak self] (action) in
            self?.recognitionType = .recognizeUnknownData
            self?.hideElements()
        }
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        
        alertController.addAction(printedTextAction)
        alertController.addAction(handWrittenAction)
        alertController.addAction(cancelAction)
        self.present(alertController, animated: true, completion: nil)

    }
    
    //MARK:- Normalize Image
    
    @IBAction func normalizeBtnClicked(_ sender: UIButton) {
        normalizeIntoLowerDimension.isUserInteractionEnabled = false
        if let image = captureImageView.image {
            if let normalizedImage = ImagePreProcessing.shared.normalizeAspectRatio(image: image) {
                UIView.transition(with: captureImageView, duration: 1.0, options: .transitionFlipFromLeft, animations: { [weak self] in
                     self?.captureImageView.image = ImagePreProcessing.shared.grayScaleImage(image: normalizedImage)
                }, completion: { [weak self] (finish) in
                    if self?.recognitionType == .toTrainANN {
                        self?.saveToTrainANN?.isHidden = true
                        self?.faceDirectionOutlet.isHidden = false
                        self?.recognizeTextBtnOutlet?.isHidden = true
                        self?.textView.isHidden = true
                    } else {
                        self?.saveToTrainANN?.isHidden = true
                        self?.faceDirectionOutlet.isHidden = true
                        self?.recognizeTextBtnOutlet?.isHidden = false
                        self?.textView.isHidden = true
                    }
                })
            }
        }
    }
    
    fileprivate func hideElements() {
        normalizeIntoLowerDimension.isHidden = true
        recognizeTextBtnOutlet.isHidden = true
        faceDirectionOutlet.isHidden = true
        saveToTrainANN.isHidden = true
        trainANNOutlet.isHidden = true
        textView.isHidden = true
    }
    
    //MARK:- To Train ANN
    
    @IBAction func faceDirectionBtnClicked(_ sender: UIButton) {
        let alertController = UIAlertController(title: "Choose Face Direction", message: "Options", preferredStyle: .actionSheet)
        TrainingANN.shared.faceDirection = [Double]()
        let leftAction = UIAlertAction(title: "Left", style: .default) { [weak self] (action) in
            TrainingANN.shared.faceDirection += FaceRecognitionDirectionConstants.left
            self?.saveToTrainANN.isHidden = false
        }
        
        let rightAction = UIAlertAction(title: "Right", style: .default) { [weak self] (action) in
            TrainingANN.shared.faceDirection += FaceRecognitionDirectionConstants.right
            self?.saveToTrainANN.isHidden = false
        }
        
        let upAction = UIAlertAction(title: "Top", style: .default) { [weak self] (action) in
            TrainingANN.shared.faceDirection += FaceRecognitionDirectionConstants.top
            self?.saveToTrainANN.isHidden = false
        }
        
        let straightAction = UIAlertAction(title: "Straight", style: .default) { [weak self] (action) in
            TrainingANN.shared.faceDirection += FaceRecognitionDirectionConstants.straight
            self?.saveToTrainANN.isHidden = false
        }
        
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        
        alertController.addAction(leftAction)
        alertController.addAction(rightAction)
        alertController.addAction(upAction)
        alertController.addAction(straightAction)
        alertController.addAction(cancelAction)
        self.present(alertController, animated: true, completion: nil)
    }
    
    //MARK:- Save into core Data
    
    @IBAction func saveToTrainANNBtnClicked(_ sender: UIButton) {
        guard let appDelegate = UIApplication.shared.delegate as? AppDelegate else { return }
        let managedContext = appDelegate.persistentContainer.viewContext
        if let entity = NSEntityDescription.entity(forEntityName: "FaceImage", in: managedContext) {
            if let faceImage = NSManagedObject(entity: entity, insertInto: managedContext) as? FaceImage {
                faceImage.setValue(TrainingANN.shared.faceDirection, forKey: "direction")
                faceImage.setValue(TrainingANN.shared.inputsTrainData, forKey: "trainData")
                
                do {
                    try managedContext.save()
                } catch let error as NSError {
                    print("Could not save. \(error), \(error.userInfo)")
                }
            }
        }
        trainANNOutlet.isHidden = false
    }
    
    //MARK:- Train the network
    
    @IBAction func trainANNBtnClicked(_ sender: UIButton) {
        TrainingANN.shared.trainBackpropagation()
    }
    
    //MARK:- Recognize
    
    @IBAction func recognize(_ sender: UIButton) {
        addActivityIndicator()
        let output = TrainingANN.shared.getOutput(inputToANN: TrainingANN.shared.inputsTrainData)
        if output.count > 0 {
            textView.isHidden = false
            textView.text = "\(output)"
        }
        removeActivityIndicator()
    }
}

//MARK:- Image picker delegate
    
extension GetImageViewController : UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        guard let image = info[UIImagePickerControllerOriginalImage] as? UIImage else { return }
        captureImageView.image = image
        captureImageView.isHidden = false
        normalizeIntoLowerDimension.isHidden = false
        normalizeIntoLowerDimension.isUserInteractionEnabled = true
        picker.dismiss(animated: true, completion: nil)
    }
}

//MARK:- Adding & removing Activity Indicator 

extension GetImageViewController {
    
    fileprivate func addActivityIndicator() {
        activityIndicator = UIActivityIndicatorView(frame: view.bounds)
        activityIndicator?.activityIndicatorViewStyle = .whiteLarge
        activityIndicator?.backgroundColor = UIColor.blue.withAlphaComponent(0.2)
        activityIndicator?.startAnimating()
        if let activityIndicator = activityIndicator {
            view.addSubview(activityIndicator)
        }
    }
    
    fileprivate func removeActivityIndicator() {
        activityIndicator?.removeFromSuperview()
        activityIndicator = nil
    }
}

