#version 460

#include "common.glsl"
#include "common-move.glsl"
#include "lygia/generative/snoise.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// Uniforms for chunk generation
uniform int chunkIndex;
uniform int chunkCount;
const float noiseScale = 0.35;

// Tree generation parameters
const float TREE_PROBABILITY = 0.05; // 1% chance per valid position
const int TREE_MIN_HEIGHT = 4 * 20; // Minimum tree height in voxels (4 blocks)
const int TREE_MAX_HEIGHT = 4 * 32; // Maximum tree height in voxels (7 blocks)
const uint WOOD_MATERIAL = 7; // Material ID for wood
const uint LEAVES_MATERIAL = 8; // Material ID for leaves

// Tree shape parameters
const int MAX_TREE_RADIUS = 4 * 10; // Maximum radius of the tree's foliage in voxels

// Root generation parameters
const float ROOT_MAX_RADIUS = 4.0 * 3.0; // Maximum radius of roots from trunk center
const float ROOT_HEIGHT = 4.0 * 2.0; // Height of root structure
const float ROOT_NOISE_SCALE = 0.8; // Scale of noise for root generation

// Convert grid coordinates to world position for noise sampling
vec3 gridToWorldPos(ivec3 gridPos) {
    return vec3(gridPos) * VOXEL_SIZE;
}

// Get world position aligned to 4x4x4 blocks for consistent terrain
vec3 getBlockAlignedPos(ivec3 gridPos) {
    vec3 worldPos = gridToWorldPos(gridPos);
    return vec3(
        floor(worldPos.x / (VOXEL_SIZE * 4.0)) * (VOXEL_SIZE * 4.0),
        floor(worldPos.y / (VOXEL_SIZE * 4.0)) * (VOXEL_SIZE * 4.0),
        floor(worldPos.z / (VOXEL_SIZE * 4.0)) * (VOXEL_SIZE * 4.0)
    );
}

// Deterministic random number generator for tree placement
float hash(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
}

// Get terrain + cave noise for solidity check using grid coordinates
bool calculateSolid(ivec3 gridPos) {
    vec3 worldBlockPos = getBlockAlignedPos(gridPos);
    float scale = noiseScale * (snoise(worldBlockPos * noiseScale * 0.1) + 1);
    
    // height map with 3 levels of noise
    float h = snoise(vec2(worldBlockPos.x, worldBlockPos.z) * scale * 0.1) * 2;
    h += snoise(vec2(worldBlockPos.x, worldBlockPos.z) * scale * 0.5) * 0.5;
    h += snoise(vec2(worldBlockPos.x, worldBlockPos.z) * scale * 0.9) * 0.25;
    h = h * 0.5 + 0.5;
    h *= 2;
    h += 6;

    // generate caves
    float cave = snoise(worldBlockPos * scale);
    return worldBlockPos.y < h && cave > 0;
}

// Get material for a position using grid coordinates
uint calculateMaterial(ivec3 gridPos) {
    vec3 worldPos = gridToWorldPos(gridPos);
    vec3 worldBlockPos = getBlockAlignedPos(gridPos);
    
    float depth = 30.0 - worldPos.y;
    float materialNoise = snoise(worldPos * noiseScale * 5) * 0.7 + 
                         snoise(worldPos * noiseScale * 10) * 0.2 + 
                         snoise(worldPos * noiseScale * 20) * 0.1;
    
    materialNoise += snoise(worldPos * noiseScale * 100) * 0.1;
    
    bool solidAbove = calculateSolid(gridPos + ivec3(0, 4, 0));
    
    if (!solidAbove) {
        return materialNoise > 0.35 ? 9 : 2;
    }
    else if (depth < 5.0 || materialNoise > 0.55) {
        return 1;
    }
    else if (materialNoise < -0.55) {
        return 4;
    }
    else {
        return 3;
    }
}

float getRootDensity(vec3 pos, vec3 basePos, float baseRadius) {
    // Get distance from center of trunk
    vec2 horizontal = pos.xz - basePos.xz;
    float distFromCenter = length(horizontal);
    
    // Get normalized height (0 at ground, 1 at top of root section)
    float heightFactor = 1.0 - (pos.y - basePos.y) / ROOT_HEIGHT;
    heightFactor = max(0.0, heightFactor);
    
    // Basic radial falloff
    float radialFalloff = 1.0 - smoothstep(baseRadius, ROOT_MAX_RADIUS, distFromCenter);
    
    // Generate noise for organic root patterns
    float rootNoise = snoise(pos * ROOT_NOISE_SCALE) * 0.5 + 0.5;
    rootNoise += snoise(pos * ROOT_NOISE_SCALE * 2.0) * 0.25;
    
    // Create finger-like root structures using angular noise
    float angle = atan(horizontal.y, horizontal.x);
    float rootPattern = abs(sin(angle * 5.0 + snoise(basePos) * 10.0));
    
    // Additional noise variation for more natural look
    float variationNoise = snoise(pos * ROOT_NOISE_SCALE * 4.0) * 0.15;
    
    // Combine all factors
    float density = radialFalloff * heightFactor * (rootPattern * 0.7 + 0.3) * (rootNoise + variationNoise);
    
    // Smooth transition near trunk
    float trunkBlend = smoothstep(baseRadius * 0.5, baseRadius * 2.0, distFromCenter);
    density = mix(1.0, density, trunkBlend);
    
    return density;
}

// Check if a position is valid for tree placement
bool canPlaceTree(ivec3 basePos) {
    // Align to 4x4 block grid
    ivec3 alignedPos = basePos - (basePos % 4);
    
    // Check if base is solid ground
    for (int x = 0; x < 4; x++) {
        for (int z = 0; z < 4; z++) {
            if (!calculateSolid(alignedPos + ivec3(x, -1, z))) return false;
        }
    }
    
    // Check if current position is air
    for (int x = 0; x < 4; x++) {
        for (int z = 0; z < 4; z++) {
            if (calculateSolid(alignedPos + ivec3(x, 0, z))) return false;
        }
    }
    
    // Check if there's enough vertical space for minimum tree height
    for (int y = 0; y < TREE_MIN_HEIGHT; y += 4) {
        for (int x = 0; x < 4; x++) {
            for (int z = 0; z < 4; z++) {
                if (calculateSolid(alignedPos + ivec3(x, y, z))) return false;
            }
        }
    }
    
    // Check if there's enough horizontal space for the tree's widest point
    for (int x = -8; x <= 8; x++) {
        for (int z = -8; z <= 8; z++) {
            if (calculateSolid(alignedPos + ivec3(x, 8, z))) return false;
        }
    }
    
    return true;
}

// Generate a tree at the given position
void generateTree(ivec3 basePos) {
    // Align to 4x4 block grid for main trunk
    ivec3 alignedPos = basePos - (basePos % 4);
    vec3 centerPos = vec3(alignedPos) + vec3(2.0, 0.0, 2.0); // Center of trunk
    
    // Use hash function for deterministic height
    float heightRandom = hash(vec3(alignedPos));
    int height = int(mix(float(TREE_MIN_HEIGHT), float(TREE_MAX_HEIGHT), heightRandom));
    height = (height / 4) * 4; // Ensure height is multiple of 4
    
    // Generate flared base and roots
    int rootRadius = int(ceil(ROOT_MAX_RADIUS));
    for(int x = -rootRadius; x <= rootRadius; x++) {
        for(int y = 0; y < int(ceil(ROOT_HEIGHT)); y++) {
            for(int z = -rootRadius; z <= rootRadius; z++) {
                ivec3 rootPos = alignedPos + ivec3(x, y, z);
                if(!isValidPosition(rootPos)) continue;
                
                // Check if we should place a root voxel
                float density = getRootDensity(vec3(rootPos), centerPos, 2.0);
                if(density > 0.5) {
                    atomicSetVoxel(rootPos, WOOD_MATERIAL);
                }
            }
        }
    }
    
    // Generate trunk (4x4)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < 4; x++) {
            for (int z = 0; z < 4; z++) {
                atomicSetVoxel(alignedPos + ivec3(x, y, z), WOOD_MATERIAL);
            }
        }
    }
    
    // Generate conical foliage
    int lowestFoliage = int(height * 0.3);
    int highestFoliage = int(height * 1.3);

    // round to nearest 4
    lowestFoliage = (lowestFoliage / 4) * 4;
    highestFoliage = (highestFoliage / 4) * 4;

    for (int y = lowestFoliage; y < highestFoliage; y += 4) {
        int radius = max(0, int((height * 1.3 - y - 1) / 8.0)); // Divide by 8 to account for 4x4 blocks
        for (int x = -radius; x <= radius; x++) {
            for (int z = -radius; z <= radius; z++) {
                // Create conical shape by checking distance from center
                float dist = dot(vec2(x, z), vec2(x, z));
                if (dist <= radius * radius) {
                    // Fill entire 4x4x4 block with leaves
                    ivec3 blockPos = alignedPos + ivec3(x * 4, y, z * 4);
                    float leafRandom = hash(vec3(blockPos));
                    if (leafRandom > 0.2) { // 80% chance for each block
                        for (int bx = 0; bx < 4; bx++) {
                            for (int by = 0; by < 4; by++) {
                                for (int bz = 0; bz < 4; bz++) {
                                    ivec3 leafPos = blockPos + ivec3(bx, by, bz);
                                    float leafVoxRandom = hash(vec3(leafPos));
                                    if (isValidPosition(leafPos) && leafVoxRandom > 0.2) { // 80% chance for each voxel
                                        atomicSetVoxel(leafPos, LEAVES_MATERIAL);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void main() {
    // Get our position in the block grid
    ivec3 blockPos = ivec3(gl_WorkGroupID) % BLOCK_SIZE;
    // Get our position in the voxel grid
    ivec3 voxelPos = ivec3(gl_LocalInvocationID);

    for (int chunk = chunkIndex; chunk < chunkIndex + chunkCount; chunk++) {
        if (chunk >= GRID_SIZE * GRID_SIZE * GRID_SIZE) break;

        ivec3 chunkPos = getChunkPos(chunk);
        
        // Calculate grid position
        ivec3 gridPos = ivec3(
            chunkPos.x * CHUNK_SIZE + blockPos.x * BLOCK_SIZE + voxelPos.x,
            chunkPos.y * CHUNK_SIZE + blockPos.y * BLOCK_SIZE + voxelPos.y,
            chunkPos.z * CHUNK_SIZE + blockPos.z * BLOCK_SIZE + voxelPos.z
        );

        if (!isValidPosition(gridPos)) continue;
        
        // Generate terrain
        if (calculateSolid(gridPos)) {
            uint materialId = calculateMaterial(gridPos);
            atomicSetVoxel(gridPos, materialId);
        }
        else {
            // Check for moss growth
            ivec3 belowPos = gridPos - ivec3(0, 1, 0);
            if (isValidPosition(belowPos) && calculateSolid(belowPos) && calculateMaterial(belowPos) == 9 && calculateMaterial(belowPos - ivec3(0, 3, 0)) == 9) {
                atomicSetVoxel(gridPos, 9);
            }
            
            // Check for tree generation only if we're at the bottom of a block
            // This ensures only one thread per block attempts tree generation
            if (voxelPos.y == 0 && voxelPos.x == 0 && voxelPos.z == 0) {
                float treeRandom = hash(vec3(gridPos));
                if (treeRandom < TREE_PROBABILITY && canPlaceTree(gridPos)) {
                    generateTree(gridPos);
                }
            }
        }
    }
}