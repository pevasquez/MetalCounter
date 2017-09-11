//
//  counter.metal
//  MetalCounter
//
//  Created by Pablo on 6/25/17.
//  Copyright © 2017 pevasquez. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void
countBlack(texture2d<float, access::read> inArray [[texture(0)]],
           volatile device uint *counter [[buffer(0)]],
           uint2 gid [[thread_position_in_grid]]) {
    
    // Atomic as we need to sync between threadgroups
    device atomic_uint *atomicBuffer = (device atomic_uint *)counter;
    float3 inColor = inArray.read(gid).rgb;
    if(inColor.r != 1.0 || inColor.g != 1.0 || inColor.b != 1.0) {
        atomic_fetch_add_explicit(atomicBuffer, 1, memory_order_relaxed);
    }
}


