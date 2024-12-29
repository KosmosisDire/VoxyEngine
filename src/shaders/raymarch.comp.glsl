#version 460

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(rgba32f, binding = 0) uniform writeonly image2D outputColor;
layout(rgba16f, binding = 1) uniform writeonly image2D outputNormals;
layout(r32f, binding = 2) uniform writeonly image2D outputDepth;
layout(r16f, binding = 3) uniform writeonly image2D outputAO;
layout(r8ui, binding = 4) uniform writeonly uimage2D outputMaterial;
layout(r16f, binding = 5) uniform writeonly image2D outputShadows;
layout(rg16f, binding = 6) uniform writeonly image2D outputMotion;

// Uniforms
uniform mat4 viewMatrix;
uniform mat4 projMatrix;
uniform vec3 cameraPos;

uniform mat4 prevViewProjMatrix;
uniform vec3 prevCameraPos;

#include "raycast.glsl"

// Calculate screen space position from world position
vec2 worldToScreen(vec3 worldPos, mat4 viewProj) {
    vec4 clipPos = viewProj * vec4(worldPos, 1.0);
    vec3 ndcPos = clipPos.xyz / clipPos.w;
    return ndcPos.xy * 0.5 + 0.5;
}

// Calculate motion vectors
vec2 calculateMotionVector(vec3 worldPos) {
    // Get current frame screen position
    vec2 currentPos = worldToScreen(worldPos, projMatrix * viewMatrix);
    
    // Get previous frame screen position
    vec2 previousPos = worldToScreen(worldPos, prevViewProjMatrix);
    
    // Calculate motion vector (in screen space)
    return previousPos - currentPos;
}


// Calculate ray direction from pixel coordinates
vec3 getRayDir(ivec2 pixel, ivec2 screenSize) {
    vec2 uv = (vec2(pixel) + 0.5) / vec2(screenSize);
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 clipPos = vec4(ndc, 1.0, 1.0);
    vec4 viewPos = inverse(projMatrix) * clipPos;
    viewPos.w = 0.0;
    vec3 worldDir = (inverse(viewMatrix) * viewPos).xyz;
    return normalize(worldDir);
}

// Calculate linear depth from world space position
float calculateLinearDepth(vec3 worldPos) {
    vec4 clipPos = projMatrix * viewMatrix * vec4(worldPos, 1.0);
    return clipPos.z / clipPos.w;
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 screenSize = imageSize(outputColor);

    if (any(greaterThanEqual(pixel, screenSize))) {
        return;
    }

    vec3 rayDir = getRayDir(pixel, screenSize);
    PixelResult result = tracePixel(cameraPos, rayDir, 512, 64);
    TraceResult primary = result.primary;
    TraceResult reflection = result.reflection;
    TraceResult transmission = result.transmission;

    vec3 color = vec3(0.0);
    uint materialId = 0;
    vec3 voxNormal = rayDir;
    vec3 faceNormal = rayDir;
    float depth = 1.0;
    float ao = 1.0;
    float sun = 1.0;
    vec2 motion = vec2(0.0);

    if (primary.hit)
    {
        materialId = primary.materialId;
        Material material = materials[materialId];

        float transDepth = (transmission.dist - primary.dist);

        if (material.reflectivity > 0.0 || material.transparency > 0.0)
        {
            float opacity = (1.0 - material.transparency);
            float reflectionLuminance = dot(result.reflectionColor, vec3(0.299, 0.587, 0.114));
            float transmissionLuminance = dot(mix(result.transparentColor, result.opaqueColor, opacity), vec3(0.299, 0.587, 0.114));
            float reflectionStrength = clamp(material.reflectivity * (reflectionLuminance / transmissionLuminance), 0.0, 1.0);

            color = mix(mix(result.transparentColor, result.opaqueColor, opacity), result.reflectionColor, reflectionStrength);
            voxNormal = mix(mix(transmission.perVoxNormal, primary.perVoxNormal, opacity), reflection.perVoxNormal, reflectionStrength);
            faceNormal = mix(mix(transmission.normal, primary.normal, opacity), reflection.normal, reflectionStrength);
            depth = calculateLinearDepth(primary.position);
            sun = mix(mix(transmission.shadowFactor, primary.shadowFactor, opacity), reflection.shadowFactor, reflectionStrength);
            ao = mix(mix(transmission.aoFactor, primary.aoFactor, opacity), reflection.aoFactor, reflectionStrength);
        }
        else
        {
            color = result.opaqueColor;
            voxNormal = primary.perVoxNormal;
            faceNormal = primary.normal;
            depth = calculateLinearDepth(primary.position);
            sun = primary.shadowFactor;
            ao = primary.aoFactor;
        }
        
        motion = calculateMotionVector(primary.position);
    } else {
        color = getSkyColor(rayDir);
        motion = calculateMotionVector(cameraPos + rayDir * 10000.0);
    }

    

    const bool useFlatNormals = false;
    vec3 useNormal = useFlatNormals ? faceNormal : voxNormal;

    // Store outputs
    imageStore(outputColor, pixel, vec4(color, 1.0));
    imageStore(outputNormals, pixel, vec4(useNormal * 0.5 + 0.5, 1.0));
    imageStore(outputDepth, pixel, vec4(depth));
    imageStore(outputAO, pixel, vec4(ao));
    imageStore(outputMaterial, pixel, uvec4(materialId));
    imageStore(outputShadows, pixel, vec4(sun));
    imageStore(outputMotion, pixel, vec4(motion, 0.0, 0.0));
}
