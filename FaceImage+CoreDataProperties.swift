//
//  FaceImage+CoreDataProperties.swift
//  FaceDirectionRecognition
//
//  Created by Ashis Laha on 21/03/17.
//  Copyright © 2017 Ashis Laha. All rights reserved.
//

import Foundation
import CoreData


extension FaceImage {

    @nonobjc public class func fetchRequest() -> NSFetchRequest<FaceImage> {
        return NSFetchRequest<FaceImage>(entityName: "FaceImage");
    }

    @NSManaged public var direction: [Double]?
    @NSManaged public var trainData: [Double]?

}
