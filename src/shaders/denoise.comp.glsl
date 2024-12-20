#version 460

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input/output textures
layout(rgba16f, binding = 1) uniform readonly image2D normalsTex;
layout(r32f, binding = 2) uniform readonly image2D depthTex;
layout(r16f, binding = 3) uniform readonly image2D rawAoTex;
layout(r16f, binding = 4) uniform image2D tempAoTex;
layout(r16f, binding = 5) uniform image2D denoisedAoTex;  // Now used as both history and output
layout(rg16f, binding = 6) uniform readonly image2D motionTex;

// Filter parameters
uniform int pass;  // 0 = temporal, 1 = horizontal blur, 2 = vertical blur
uniform float filterSize;

// Constants
const float DEPTH_THRESHOLD = 0.1;
const float NORMAL_THRESHOLD = 0.1;
const int MAX_RADIUS = 2;
const float MIN_BLEND_FACTOR = 0.1;
const float MAX_BLEND_FACTOR = 0.9;

// Helper function for bilateral depth weight
float getDepthWeight(float centerDepth, float sampleDepth) {
    // Convert from normalized depth [0,1] to view space Z
    // Assuming reversed-Z projection where 1.0 is near and 0.0 is far
    float near = 0.01; // Adjust these based on your camera settings
    float far = 500.0;
    
    float centerZ = -near * far / (centerDepth * (far - near) - far);
    float sampleZ = -near * far / (sampleDepth * (far - near) - far);
    
    // Compare in linear space with relative threshold
    float relativeThreshold = 0.1; // 10% relative difference threshold
    float relativeDiff = abs(centerZ - sampleZ) / min(abs(centerZ), abs(sampleZ));
    
    return exp(-relativeDiff * relativeDiff / (relativeThreshold * relativeThreshold));
}

float getNormalWeight(vec3 centerNormal, vec3 sampleNormal) {
    float dotProduct = max(0.0, dot(centerNormal, sampleNormal));
    // More aggressive normal weight - higher power means sharper normal edges
    return pow(dotProduct, 8.0);
}

// Spatial filter processing (same as before)
vec2 processLine(ivec2 pixel, ivec2 dir, int radius, float centerDepth) {
    float sum = 0.0;
    float totalWeight = 0.0;
    
    // Get center normal
    vec3 centerNormal = imageLoad(normalsTex, pixel).rgb * 2.0 - 1.0;
    float centerValue = imageLoad(tempAoTex, pixel).r;
    
    sum = centerValue;
    totalWeight = 1.0;
    
    for (int i = -radius; i <= radius; i++) {
        if (i == 0) continue;
        
        ivec2 samplePos = pixel + dir * i;
        
        if (any(lessThan(samplePos, ivec2(0))) || 
            any(greaterThanEqual(samplePos, imageSize(rawAoTex)))) {
            continue;
        }
        
        float sampleDepth = imageLoad(depthTex, samplePos).r;
        float sampleValue = imageLoad(tempAoTex, samplePos).r;
        vec3 sampleNormal = imageLoad(normalsTex, samplePos).rgb * 2.0 - 1.0;
        
        // Combine depth and normal weights
        float w = getDepthWeight(centerDepth, sampleDepth);
        w *= getNormalWeight(centerNormal, sampleNormal);
        
        float dist = float(i) / float(radius);
        w *= exp(-dist * dist * 2.0);
        
        sum += sampleValue * w;
        totalWeight += w;
    }
    
    return vec2(sum, totalWeight);
}


// New temporal reprojection function
float reprojectHistory(ivec2 pixel, out float validityWeight) {
    vec2 motion = imageLoad(motionTex, pixel).xy;
    vec2 prevPos = vec2(pixel) + motion * vec2(imageSize(rawAoTex));
    ivec2 prevPixel = ivec2(prevPos);
    
    if (any(lessThan(prevPixel, ivec2(0))) || 
        any(greaterThanEqual(prevPixel, imageSize(denoisedAoTex)))) {
        validityWeight = 0.0;
        return 0.0;
    }
    
    float depth = imageLoad(depthTex, pixel).r;
    vec3 normal = imageLoad(normalsTex, pixel).rgb * 2.0 - 1.0;
    
    float prevDepth = imageLoad(depthTex, prevPixel).r;
    vec3 prevNormal = imageLoad(normalsTex, prevPixel).rgb * 2.0 - 1.0;
    
    float depthWeight = getDepthWeight(depth, prevDepth);
    // More aggressive normal weight for temporal pass
    float normalWeight = pow(max(0.0, dot(normal, prevNormal)), 16.0);
    validityWeight = depthWeight * normalWeight;
    
    return imageLoad(denoisedAoTex, prevPixel).r;
}

void main()
{
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(rawAoTex);
    
    if (any(greaterThanEqual(pixel, size))) {
        return;
    }

    float depth = imageLoad(depthTex, pixel).r;
    
    // Skip far plane
    if (depth >= 0.9999) {
        float value = 1.0;
        if (pass == 1) imageStore(tempAoTex, pixel, vec4(value));
        else imageStore(denoisedAoTex, pixel, vec4(value));
        return;
    }
    
    if (pass == 0) {
        // Temporal pass
        float currentValue = imageLoad(rawAoTex, pixel).r;
        float historyValidity;
        float historyValue = reprojectHistory(pixel, historyValidity);
        
        // Calculate temporal blend factor
        float blendFactor = mix(MIN_BLEND_FACTOR, MAX_BLEND_FACTOR, historyValidity);
        
        // Blend current and history
        float result = mix(currentValue, historyValue, blendFactor);
        
        // Store result in temp buffer for spatial filtering
        imageStore(tempAoTex, pixel, vec4(result));
    }
    else {
        // Spatial filtering passes
        int radius = int(filterSize * float(MAX_RADIUS));
        vec2 result;
        
        if (pass == 1) {
            // Horizontal pass
            result = processLine(pixel, ivec2(1, 0), radius, depth);
            float filtered = result.x / max(result.y, 0.0001);
            imageStore(tempAoTex, pixel, vec4(filtered));
        }
        else {
            // Vertical pass
            result = processLine(pixel, ivec2(0, 1), radius, depth);
            float filtered = result.x / max(result.y, 0.0001);
            imageStore(denoisedAoTex, pixel, vec4(filtered));
        }
    }
}