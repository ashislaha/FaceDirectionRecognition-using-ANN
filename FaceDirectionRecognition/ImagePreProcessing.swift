//
//  ImagePreProcessing.swift
//  Handwritten Recognition
//
//  Created by Ashis Laha on 07/03/17.
//  Copyright © 2017 Ashis Laha. All rights reserved.
//

import Foundation
import CoreGraphics
import UIKit

struct ImageProcessingConstants {
    static let maxDimension : CGFloat  = 100
    static let optimalDimensionX : Int = 50
    static let optimalDimensionY : Int = 45
    static let suppressionMinIntensity : Double = 50
    static let suppressionMaxIntensity : Double = 150
}

struct PixelData {
    var r : UInt8 = 0
    var g : UInt8 = 0
    var b : UInt8 = 0
    var a : UInt8 = 0
}

class ImagePreProcessing {
    
    /*
     
     Image Preprocessing : 
        
        1. Normalize Aspect ratio with respect to N*M dimension
        2. Edge detection of the image
        3. Reduce the noise from image
     
     */
    
    static let shared = ImagePreProcessing()
    var optimalPixelMatrix = [[Double]]()
    var pixelData = [[PixelData]]()
    
    fileprivate func mask8( x : UInt32) -> UInt32 { return x & 0xFF }
    fileprivate func R(x : UInt32) -> UInt32 { return mask8(x: x) }
    fileprivate func G(x : UInt32) -> UInt32 { return mask8(x: x >> 8 )}
    fileprivate func B(x : UInt32) -> UInt32 { return mask8(x: x >> 16)}
    fileprivate func alphaComponent(x : UInt32) -> UInt32 { return mask8(x: x>>24)}
    fileprivate func RGBAlphaMake( r : UInt32, g : UInt32, b : UInt32, alpha : UInt32 ) -> UInt32 { return mask8(x: r) | mask8(x: g<<8) | mask8(x: b<<16) | mask8(x: alpha<<24)}

    
    //MARK:- Normalize Aspect ratio with respect to ImageProcessingConstants.maxDimension
    
    func normalizeAspectRatio(image : UIImage) -> UIImage? {
        
        let width = image.size.width
        let height = image.size.height
        
        var scaledSize = CGSize(width: ImageProcessingConstants.maxDimension, height: ImageProcessingConstants.maxDimension)
        var scaleFactor : CGFloat = 0
        
        if width > height {
            scaleFactor = height / width
            scaledSize.height = ImageProcessingConstants.maxDimension * scaleFactor
        } else {
            scaleFactor = width / height
            scaledSize.width = ImageProcessingConstants.maxDimension * scaleFactor
        }
        
        UIGraphicsBeginImageContext(scaledSize)
        image.draw(in: CGRect(origin: CGPoint.zero, size: scaledSize))
        guard let scaledImage = UIGraphicsGetImageFromCurrentImageContext() else { return nil }
        UIGraphicsEndImageContext()
        return scaledImage
    }
    
    //MARK:- GrayScale Image
    
    func grayScaleImage(image : UIImage) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(image.size, true, 1.0)
        let rect = CGRect(origin: CGPoint.zero, size: image.size)
        image.draw(in: rect, blendMode: .luminosity, alpha: 1.0)
        let returnImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        if let image = returnImage {
            TrainingANN.shared.inputsTrainData =  optimalIntensityMatrix(image: image)
        }
        return returnImage
    }
    
    
    fileprivate func optimalIntensityMatrix(image: UIImage) -> [Double] {
       
        // pixel calculation  , logMatrix is 100*75
        
        var logMatrix = logPixelOfImage(image: image)
        optimalPixelMatrix = [[Double]]()
        
        // assignment
        for i in 0..<ImageProcessingConstants.optimalDimensionX {
            var temp = [Double]()
            for j in 0..<ImageProcessingConstants.optimalDimensionY {
                temp.append(logMatrix[i][j])
            }
            optimalPixelMatrix.append(temp)
        }
        
        printMatrix(inputMatrix: optimalPixelMatrix)
        
        // Edge detection 
        
        let cannyResults = cannyEdgeDetectionOperator(inputMatrix: optimalPixelMatrix)
        
        // calculate optimal martix of [optimalDimensionX * optimalDimensionY]
        
        return calculateInputFromOptimalIntensityMatrix(inputMatrix: cannyResults)
    }

    fileprivate func calculateInputFromOptimalIntensityMatrix(inputMatrix : [[Double]]) -> [Double] {
        var intensities = [Double]()
        for i in 0..<ImageProcessingConstants.optimalDimensionX {
            for j in 0..<ImageProcessingConstants.optimalDimensionY {
                intensities.append(inputMatrix[i][j])
            }
        }
        return intensities
    }
    
    //MARK:- Log Pixel image 
    
    fileprivate func logPixelOfImage(image : UIImage) -> [[Double]] {
        guard let coreImageRef = image.cgImage else { return [] }
        let height = coreImageRef.height
        let width = coreImageRef.width
        
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        let pixels  = UnsafeMutablePointer<UInt32>.allocate(capacity: height * width)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixels, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace,bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        context?.draw(coreImageRef, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // print the image
        
        var pixelMatrix = [[Double]]()
        
        var tempPixels = pixels
        for i in 0..<height {
            var rowData = [Double]()
            var rowPixel = [PixelData]()
            
            for j in 0..<width {
                if i < ImageProcessingConstants.optimalDimensionX && j < ImageProcessingConstants.optimalDimensionY {
                    let color = tempPixels[j + i * width]
                    let doubleValue = Double(R(x: color)+G(x: color)+B(x: color))/3.0
                    rowData.append(doubleValue)
                    
                    // pixel data collection
                    rowPixel.append(PixelData(r: UInt8(R(x: color)), g: UInt8(G(x: color)), b: UInt8(B(x: color)), a: UInt8(alphaComponent(x : color))))
                }
                // increment the pointer
                tempPixels = tempPixels + 1
            }
            
            if rowData.count > 0  {  pixelMatrix.append(rowData) }
            if rowPixel.count > 0 {  pixelData.append(rowPixel) }
        }
        
        TrainingANN.shared.pixels = pixelData
        free(pixels)
        
        return pixelMatrix
    }
    
    //MARK:- Canny Edge Dectection Algo.
    
    fileprivate func cannyEdgeDetectionOperator(inputMatrix : [[Double]]) -> [[Double]] {
        
        // apply gaussian filter to remove noise
        let gaussainOutput  = gaussianFilter(inputMatrix:inputMatrix)
        printMatrix(inputMatrix: gaussainOutput)
        
        // Find the intensity gradients of the image (apply sobel operator)
        let sobelOutput = sobelOperator(inputMatrix: gaussainOutput)
        printMatrix(inputMatrix: sobelOutput)
        
        // Apply non-maximum suppression to get rid of spurious response to edge detection
        let suppressedMatrix = suppression(inputMatrix: sobelOutput, suppressionMin: ImageProcessingConstants.suppressionMinIntensity, suppressionMax: ImageProcessingConstants.suppressionMaxIntensity)
        printMatrix(inputMatrix: suppressedMatrix)
        
        return suppressedMatrix
    }
    
    
    //MARK:- Remove noise from Image
    
    func gaussianFilter(inputMatrix : [[Double]]) -> [[Double]] { // use 5*5 filter 
        
        let gaussainMatrix = [
                                [ 2.0, 4.0, 5.0, 4.0, 2.0 ],
                                [ 4.0, 9.0, 12.0, 9.0, 4.0],
                                [ 5.0, 12.0, 15.0, 12.0, 5.0],
                                [ 4.0, 9.0, 12.0, 9.0, 4.0],
                                [ 2.0, 4.0, 5.0, 4.0, 2.0 ]
                            ]
        var filteredMatrix = [[Double]]()
        
        // intialize filteredMatrix 
        
        for _ in 0..<ImageProcessingConstants.optimalDimensionX {
            var temp = [Double]()
            for _ in 0..<ImageProcessingConstants.optimalDimensionY {
                temp.append(0.0)
            }
            filteredMatrix.append(temp)
        }
        
        // compute
        
        for x in 2..<ImageProcessingConstants.optimalDimensionX-2 {
            for y in 2..<ImageProcessingConstants.optimalDimensionY-2 {
                
                var totalIntensity = 0.0
                
                var xDelta = -2
                for i in 0..<5 {
                    var yDelta = -2
                    for j in 0..<5 {
                        totalIntensity += gaussainMatrix[i][j] * inputMatrix[x+xDelta][y+yDelta]
                        yDelta += 1
                    }
                    xDelta += 1
                }
                
                filteredMatrix[x][y] = totalIntensity / 159.0
            }
        }
        return filteredMatrix
    }
    

    //MARK:- Edge Detection Technique ( Sobel operator ) on optimalPixelMatrix
    
    fileprivate func sobelOperator(inputMatrix : [[Double]]) -> [[Double]] {
        
        var filteredOutput = [[Double]]()
        
        // Initialize all elements with Zero elements 
        
        for _ in 0..<ImageProcessingConstants.optimalDimensionX {
            var tempArr = [Double]()
            for _ in 0..<ImageProcessingConstants.optimalDimensionY {
                tempArr.append(0)
            }
            filteredOutput.append(tempArr)
        }
        
        // Calculate with Sobel fileter (3*3 matrix )
        
        let sobelFilterHorizonal : [[Double]] = [[1,0,-1],[2,0,-2],[1,0,-1]]
        let sobelFilterVertical  : [[Double]] = [[1,2,1],[0,0,0],[-1,-2,-1]]
        
        // apply sobel filter 9-Neighbor
        
        for i in 1..<ImageProcessingConstants.optimalDimensionX-1 {
            for j in 1..<ImageProcessingConstants.optimalDimensionY-1 {
                
                let Gx = (inputMatrix[i-1][j-1] * sobelFilterHorizonal[0][0] + inputMatrix[i-1][j+1] * sobelFilterHorizonal[0][2] +
                          inputMatrix[i][j-1] * sobelFilterHorizonal[1][0] + inputMatrix[i][j+1] * sobelFilterHorizonal[1][2] +
                          inputMatrix[i+1][j-1] * sobelFilterHorizonal[2][0] + inputMatrix[i+1][j+1] * sobelFilterHorizonal[2][2]
                        )
                
                let Gy = (inputMatrix[i-1][j-1] * sobelFilterVertical[0][0] + inputMatrix[i-1][j] * sobelFilterVertical[0][1] +
                    inputMatrix[i-1][j+1] * sobelFilterVertical[0][2] + inputMatrix[i+1][j-1] * sobelFilterVertical[2][0] +
                    inputMatrix[i+1][j] * sobelFilterVertical[2][1] + inputMatrix[i+1][j+1] * sobelFilterVertical[2][2])
                
                filteredOutput[i][j] = sqrt(Gx*Gx+Gy*Gy)
            }
        }
        return filteredOutput
    }
    
    //MARK:- suppression
    
    
    fileprivate func suppression(inputMatrix : [[Double]], suppressionMin : Double , suppressionMax : Double) -> [[Double]] {
        
        var results = [[Double]]()
        
        //initialise
        for _ in 0..<ImageProcessingConstants.optimalDimensionX {
            var tempArr = [Double]()
            for _ in 0..<ImageProcessingConstants.optimalDimensionY {
                tempArr.append(0)
            }
            results.append(tempArr)
        }
        
        // supress
        for row in 0..<ImageProcessingConstants.optimalDimensionX {
            for column in 0..<ImageProcessingConstants.optimalDimensionY {
                let input = inputMatrix[row][column]
                if input > suppressionMax || input < suppressionMin {
                    results[row][column] = 0.0
                } else {
                    results[row][column] = input
                }
            }
        }
        return results
    }
    
    //MARK:- Create Image from Pixel data
    
    fileprivate func createImageFromRGBData(inputMatrix : [Double], width : Int, height : Int) -> UIImage {
        
        var resultImage = UIImage()
        let bytesPerPixel = 4  // 4 bytes or 32 bits
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8 // rgba, so total lenghth = 32 bits
        
        var pixelData = inputMatrix
        
        let bitmapContext = CGContext(data: &pixelData[0], width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        if let cgImage = bitmapContext?.makeImage() {
            let image = UIImage(cgImage: cgImage)
            resultImage = image
        }
        return resultImage
    }
    
    //MARK:- Print Matrix
    
    fileprivate func printMatrix(inputMatrix : [[Double]]) {
        print("\n\n *********** PRINT MATRIX ********** \n\n")
        
        for i in 0..<ImageProcessingConstants.optimalDimensionX {
            for j in 0..<ImageProcessingConstants.optimalDimensionY {
                let string = String(format: "%4.0f", inputMatrix[i][j])
                print(string, terminator : "")
            }
            print("")
        }
    }

    
}
