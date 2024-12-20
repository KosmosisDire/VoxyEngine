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
uniform vec3 sunDir;
uniform vec3 sunColor;
uniform vec3 cameraPos;
uniform mat4 viewMatrix;
uniform mat4 projMatrix;

// Indirect lighting parameters
const vec3 ambientColor = vec3(0.84, 1, 0.98);
const float ambientStrength = 0.4;

// Tone mapping parameters
const float exposure = 1.1;

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
    vec3 normal = imageLoad(normalsTex, pixel).xyz * 2.0 - 1.0;
    float depth = imageLoad(depthTex, pixel).r;
    float ao = imageLoad(aoTex, pixel).r;
    float shadow = imageLoad(shadowTex, pixel).r;


    vec3 specular = vec3(0.0);
    float NdotL = 0.0;
    
    vec3 finalColor;
    
    if (depth >= 0.9999) {
        finalColor = albedo;
    } else {
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
        NdotL = clamp(dot(normal, sunDir) + material.transparency, 0.0, 1.0);
    
        
        // Calculate specular lighting
        float shininess = material.shininess * 100;
        float specularStrength = material.specularStrength * 10;
        vec3 reflectDir = reflect(-sunDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
        specular = specularStrength * spec * sunColor;
        
        // Combine all lighting components
        vec3 directLight = (sunColor + specular) * shadow * NdotL;
        vec3 indirectLight = ambientColor * ambientStrength * ao;
        
        // Final color calculation
        finalColor = albedo * (directLight + indirectLight);
        
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