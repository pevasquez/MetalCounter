//
//  main.swift
//  MetalCounter
//
//  Created by Pablo on 6/25/17.
//  Copyright © 2017 pevasquez. All rights reserved.
//

import Foundation
import Metal
import CoreGraphics
import AppKit

guard FileManager.default.fileExists(atPath: "./testImage.png") else {
    print("./testImage.png does not exist")
    exit(1)
}

let url = URL(fileURLWithPath: "./testImage.png")
let imageData = try Data(contentsOf: url)

guard let image = NSImage(data: imageData),
    let imageRef = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    print("Failed to load image data")
    exit(1)
}

let bytesPerPixel = 4
let bytesPerRow = bytesPerPixel * imageRef.width

var rawData = [UInt8](repeating: 0, count: Int(bytesPerRow * imageRef.height))

let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue).union(.byteOrder32Big)
let colorSpace = CGColorSpaceCreateDeviceRGB()

let context = CGContext(data: &rawData,
                        width: imageRef.width,
                        height: imageRef.height,
                        bitsPerComponent: 8,
                        bytesPerRow: bytesPerRow,
                        space: colorSpace,
                        bitmapInfo: bitmapInfo.rawValue)

let fullRect = CGRect(x: 0, y: 0, width: CGFloat(imageRef.width), height: CGFloat(imageRef.height))
context?.draw(imageRef, in: fullRect, byTiling: false)

// Get access to iPhone or iPad GPU
guard let device = MTLCreateSystemDefaultDevice() else {
    exit(1)
}

let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
    pixelFormat: .rgba8Unorm,
    width: Int(imageRef.width),
    height: Int(imageRef.height),
    mipmapped: true)

let texture = device.makeTexture(descriptor: textureDescriptor)

let region = MTLRegionMake2D(0, 0, Int(imageRef.width), Int(imageRef.height))
texture.replace(region: region, mipmapLevel: 0, withBytes: &rawData, bytesPerRow: Int(bytesPerRow))

// Queue to handle an ordered list of command buffers
let commandQueue = device.makeCommandQueue()

// Buffer for storing encoded commands that are sent to GPU
let commandBuffer = commandQueue.makeCommandBuffer()

// Access to Metal functions that are stored in Shaders.metal file, e.g. sigmoid()
guard let defaultLibrary = device.makeDefaultLibrary() else {
    print("Failed to create default metal shader library")
    exit(1)
}

// Encoder for GPU commands
let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()

// hardcoded to 16 for now (recommendation: read about threadExecutionWidth)
var threadsPerGroup = MTLSize(width:16, height:16, depth:1)
var numThreadgroups = MTLSizeMake(texture.width / threadsPerGroup.width,
                                  texture.height / threadsPerGroup.height,
                                  1);

// b. set up a compute pipeline with Sigmoid function and add it to encoder
let countBlackProgram = defaultLibrary.makeFunction(name: "countBlack")
let computePipelineState = try device.makeComputePipelineState(function: countBlackProgram!)
computeCommandEncoder.setComputePipelineState(computePipelineState)


// set the input texture for the countBlack() function, e.g. inArray
// atIndex: 0 here corresponds to texture(0) in the countBlack() function
computeCommandEncoder.setTexture(texture, index: 0)

// create the output vector for the countBlack() function, e.g. counter
// atIndex: 1 here corresponds to buffer(0) in the Sigmoid function
var counterBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                        options: .storageModeShared)
computeCommandEncoder.setBuffer(counterBuffer, offset: 0, index: 0)

computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

computeCommandEncoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

// a. Get GPU data
// outVectorBuffer.contents() returns UnsafeMutablePointer roughly equivalent to char* in C
var data = NSData(bytesNoCopy: counterBuffer.contents(),
                  length: MemoryLayout<UInt32>.size,
                  freeWhenDone: false)
// b. prepare Swift array large enough to receive data from GPU
var finalResultArray = [UInt32](repeating: 0, count: 1)

// c. get data from GPU into Swift array
data.getBytes(&finalResultArray, length: MemoryLayout<UInt>.size)

print("Found \(finalResultArray[0]) non-white pixels")

// d. YOU'RE ALL SET!
