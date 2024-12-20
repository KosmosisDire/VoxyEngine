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
uniform float time;

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

// Helper function for Hammersley sequence
float radicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}
// Frisvad's method for stable orthonormal basis
void createCoordinateSystem(in vec3 N, out vec3 T1, out vec3 T2) {
    if (abs(N.x) > abs(N.y)) {
        float invLen = 1.0 / sqrt(N.x * N.x + N.z * N.z);
        T1 = vec3(-N.z * invLen, 0.0, N.x * invLen);
    } else {
        float invLen = 1.0 / sqrt(N.y * N.y + N.z * N.z);
        T1 = vec3(0.0, N.z * invLen, -N.y * invLen);
    }
    T2 = cross(N, T1);
}

// Improved distance falloff function
float aoFalloff(float dist, float maxDist) {
    float x = dist / maxDist;
    // Smooth cubic falloff
    return 1.0 - smoothstep(0.0, 1.0, x * x * (3.0 - 2.0 * x));
}

// High quality procedural noise function
float hash3D(vec3 p) {
    p = fract(p * vec3(443.8975, 397.2973, 491.1871));
    p += dot(p.zxy, p.yxz + 19.19);
    return fract(p.x * p.y * p.z);
}

// Interleaved gradient noise - provides good spatial distribution
float interleavedGradientNoise(vec2 coord, float temporal) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(coord + temporal * vec2(47.0, 17.0), magic.xy)));
}

float spatioTemporalBlueNoise(vec2 coord, float index, float temporal) {
    // Combine different noise patterns for better distribution
    float noise1 = hash3D(vec3(coord * 0.017, index + temporal * 0.031));
    float noise2 = interleavedGradientNoise(coord, temporal + index * 0.073);
    
    // Weighted combination
    return mix(noise1, noise2, 0.5);
}

// Modified shadow ray check to work with accumulator
float castShadowRay(vec3 origin, vec3 normal) {
    origin += normal * EPSILON;
    TransparencyAccumulator shadowAcc = raymarchMultiLevel(origin, sunDir, 512);
    return 1.0 - shadowAcc.opacity;  // Use opacity for shadow calculation
}

// Modified AO calculation to work with accumulator
float calculateAO(vec3 position, vec3 normal, int sampleCount, vec2 pixelCoord) {
    float ao = 0.0;
    uint seed = uint(pixelCoord.x * 123745.0 + pixelCoord.y * 677890.0 + time * 111111.0);
    
    // Constants remain the same...
    const float MAX_DISTANCE = VOXEL_SIZE * 8;
    const float GOLDEN_RATIO = 1.618033988749895;
    float sqrtSamples = sqrt(float(sampleCount));
    float invSqrtSamples = 1.0 / sqrtSamples;
    
    // Create tangent space...
    vec3 tangent, bitangent;
    createCoordinateSystem(normal, tangent, bitangent);
    
    for (int i = 0; i < sampleCount; i++) {
        // Sample generation remains the same...
        float strataU = float(i % int(sqrtSamples)) * invSqrtSamples;
        float strataV = float(i / int(sqrtSamples)) * invSqrtSamples;
        
        float t = float(i) + 0.5;
        float phi = t * 2.0 * PI * GOLDEN_RATIO;
        
        vec2 jitter = vec2(
            spatioTemporalBlueNoise(pixelCoord, float(i), time),
            spatioTemporalBlueNoise(pixelCoord + vec2(117.0, 313.0), float(i), time)
        ) * 100.0 * invSqrtSamples;
        
        float u = fract(t / float(sampleCount) + jitter.x);
        float v = fract(phi / (2.0 * PI) + jitter.y);
        
        float cosTheta = sqrt(1.0 - u);
        float sinTheta = sqrt(u);
        float phi_final = v * 2.0 * PI;
        
        vec3 sampleDir = vec3(
            cos(phi_final) * sinTheta,
            sin(phi_final) * sinTheta,
            cosTheta
        );
        
        sampleDir = normalize(
            tangent * sampleDir.x +
            bitangent * sampleDir.y +
            normal * sampleDir.z
        );
        
        // Modified raymarch call to use accumulator
        TransparencyAccumulator aoAcc = raymarchMultiLevel(
            position + normal * (EPSILON * 2.0),
            sampleDir,
            8
        );
        
        // Use first hit distance from accumulator if available
        float dist = aoAcc.hasHit ? aoAcc.hits[0].dist : MAX_DISTANCE;
        
        if (dist >= 0.0) {
            float weight = aoFalloff(dist, MAX_DISTANCE);
            ao += weight * aoAcc.opacity;
        }
    }
    
    float aoValue = 1.0 - (ao / float(sampleCount));
    return clamp(pow(aoValue, 1.2), 0.0, 1.0);
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
    TransparencyAccumulator result = raymarchMultiLevel(cameraPos, rayDir, 512, 64);

    // Initialize outputs with default values
    vec3 color = vec3(0.0);
    uint materialId = 0;
    vec3 voxNormal = rayDir;
    vec3 faceNormal = rayDir;
    float depth = 1.0;
    float ao = 1.0;
    float sun = 1.0;
    vec2 motion = vec2(0.0);
    
    if (result.hasHit) {
        // Use first hit for primary surface information
        RaymarchHit primaryHit = result.hits[0];
        RaymarchHit secondHit = result.numHits > 1 ? result.hits[1] : result.hits[0];

        // Get accumulated color including transparency
        color = result.color;
        
        // Get material ID from first hit
        materialId = primaryHit.materialId;
        
        // Get normals from first hit
        faceNormal = primaryHit.normal;
        voxNormal = primaryHit.perVoxNormal;
        
        // Calculate depth from first hit position
        depth = calculateLinearDepth(primaryHit.position);
        
        // Calculate AO using first hit information
        ao = 1.0; //calculateAO(primaryHit.position, faceNormal, 10, vec2(pixel));
        
        // Calculate shadow using first hit information
        float sunCheck = max(0.0, dot(voxNormal, sunDir));
        if (sunCheck > 0)
        {
            sun *= castShadowRay(primaryHit.position, faceNormal);
        }
        
        // Calculate motion vectors using first hit position
        motion = calculateMotionVector(primaryHit.position);
    }
    else {
        // For sky pixels
        color = getSkyColor(rayDir);
        
        // Calculate motion for sky using a far point
        vec3 farPoint = cameraPos + rayDir * 10000.0;
        motion = calculateMotionVector(farPoint);
    }

    // Store outputs
    imageStore(outputColor, pixel, vec4(color, 1.0));
    imageStore(outputNormals, pixel, vec4(voxNormal * 0.5 + 0.5, 1.0));
    imageStore(outputDepth, pixel, vec4(depth));
    imageStore(outputAO, pixel, vec4(ao));
    imageStore(outputMaterial, pixel, uvec4(materialId));
    imageStore(outputShadows, pixel, vec4(sun));
    imageStore(outputMotion, pixel, vec4(motion, 0.0, 0.0));
}