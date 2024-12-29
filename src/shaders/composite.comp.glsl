#version 460

#include "common.glsl"

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input textures
layout(rgba32f, binding = 0) uniform readonly image2D colorTex;
layout(rgba16f, binding = 1) uniform readonly image2D normalsTex;
layout(r32f, binding = 2) uniform readonly image2D depthTex;
layout(r16f, binding = 3) uniform readonly image2D aoTex;
layout(r8ui, binding = 4) uniform readonly uimage2D materialTex;
layout(r16f, binding = 5) uniform readonly image2D shadowTex;

// Output texture
layout(rgba8, binding = 6) uniform writeonly image2D outputTex;

// Lighting uniforms
uniform vec3 cameraPos;
uniform mat4 viewMatrix;
uniform mat4 projMatrix;

#include "atmosphere.glsl"

// Normal blur parameters
const float NORMAL_BLUR_RADIUS = 0.0;
const float NORMAL_BLUR_SIGMA = 2.0;
const float DEPTH_SENSITIVITY = 10000.0;  // Adjust to control how much depth differences affect the blur

// Tone mapping parameters
const float exposure = 1.3;

const int CROSSHAIR_SIZE = 10;
const int CROSSHAIR_THICKNESS = 2;

// ACES tone mapping curve
vec3 ACESFilm(vec3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

bool isPartOfCrosshair(ivec2 pixel, ivec2 screenSize) {
    ivec2 center = screenSize / 2;
    
    bool horizontalPart = (
        abs(pixel.y - center.y) < CROSSHAIR_THICKNESS &&
        abs(pixel.x - center.x) < CROSSHAIR_SIZE
    );
    
    bool verticalPart = (
        abs(pixel.x - center.x) < CROSSHAIR_THICKNESS &&
        abs(pixel.y - center.y) < CROSSHAIR_SIZE
    );
    
    return horizontalPart || verticalPart;
}

float getLinearDepth(float depth, mat4 proj) {
    float near = proj[3][2] / (proj[2][2] - 1.0);
    float far = proj[3][2] / (proj[2][2] + 1.0);
    return (2.0 * near) / (far + near - depth * (far - near));
}

vec3 sampleNormalWithOffset(ivec2 pixel, ivec2 offset, ivec2 screenSize) {
    ivec2 samplePos = clamp(pixel + offset, ivec2(0), screenSize - ivec2(1));
    return imageLoad(normalsTex, samplePos).xyz * 2.0 - 1.0;
}

float sampleDepthWithOffset(ivec2 pixel, ivec2 offset, ivec2 screenSize) {
    ivec2 samplePos = clamp(pixel + offset, ivec2(0), screenSize - ivec2(1));
    return imageLoad(depthTex, samplePos).r;
}

vec3 blurNormal(ivec2 pixel, ivec2 screenSize) {
    vec3 blurredNormal = vec3(0.0);
    float totalWeight = 0.0;
    
    // Get center depth
    float centerDepth = getLinearDepth(imageLoad(depthTex, pixel).r, projMatrix);
    
    // Use the actual radius parameter
    int radius = int(ceil(NORMAL_BLUR_RADIUS));
    
    // Sample in a square with the specified radius
    for(int x = -radius; x <= radius; x++) {
        for(int y = -radius; y <= radius; y++) {
            vec2 offset = vec2(x, y);
            float distance = length(offset);
            
            // Skip samples outside the circular radius
            if (distance > NORMAL_BLUR_RADIUS) continue;
            
            // Get sample depth
            float sampleDepth = getLinearDepth(sampleDepthWithOffset(pixel, ivec2(x, y), screenSize), projMatrix);
            
            // Calculate depth difference weight
            float depthDiff = abs(centerDepth - sampleDepth);
            float depthWeight = exp(-depthDiff * DEPTH_SENSITIVITY);
            
            // Combine spatial and depth weights
            float spatialWeight = exp(-(distance * distance) / (2.0 * NORMAL_BLUR_SIGMA * NORMAL_BLUR_SIGMA));
            float weight = spatialWeight * depthWeight;
            
            vec3 sampleNormal = sampleNormalWithOffset(pixel, ivec2(x, y), screenSize);
            
            blurredNormal += sampleNormal * weight;
            totalWeight += weight;
        }
    }
    
    // Normalize both the weight and the vector
    return normalize(blurredNormal / totalWeight);
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 screenSize = imageSize(colorTex);
    
    if (any(greaterThanEqual(pixel, screenSize))) {
        return;
    }
    
    // Load all input textures
    vec3 albedo = imageLoad(colorTex, pixel).rgb;
    int materialId = int(imageLoad(materialTex, pixel).r);
    Material material = materials[materialId];
    vec3 normal = blurNormal(pixel, screenSize);  // Using blurred normal
    float depth = imageLoad(depthTex, pixel).r;
    float ao = imageLoad(aoTex, pixel).r;
    float shadow = imageLoad(shadowTex, pixel).r;

    vec3 finalColor;
    float NdotL;
    vec3 specular;

    if (depth >= 0.9999) {
        finalColor = albedo;
    }
    else 
    {
        // Calculate view direction
        vec4 ndc = vec4(
            (2.0 * pixel.x / float(screenSize.x) - 1.0),
            (1.0 - 2.0 * pixel.y / float(screenSize.y)),
            2.0 * depth - 1.0,
            1.0
        );
        
        vec4 worldPos = inverse(projMatrix * viewMatrix) * ndc;
        worldPos /= worldPos.w;
        vec3 viewDir = normalize(cameraPos - worldPos.xyz);
       
        // Calculate direct diffuse lighting
        NdotL = clamp(dot(normal, sunDir), 0.0, 1.0);
    
        // Calculate specular lighting
        float shininess = material.shininess * 100;
        float specularStrength = material.specularStrength * 10;
        vec3 reflectDir = reflect(-sunDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
        specular = specularStrength * spec * calculateSunColor(sunDir);

        // Final color calculation
        finalColor = (albedo + specular) * (shadow + 0.5 / 2.0) * ao;
        
        // Apply tone mapping
        finalColor *= exposure;
        finalColor = ACESFilm(finalColor);
    }
    
    // Apply crosshair if pixel is part of it
    if (isPartOfCrosshair(pixel, screenSize)) {
        finalColor = 1.0 - finalColor;
    }
    
    // Output final LDR color
    imageStore(outputTex, pixel, vec4(vec3(finalColor), 1.0));
}