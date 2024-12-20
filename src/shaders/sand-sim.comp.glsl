#version 460

#include "common.glsl"
#include "common-move.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// Uniforms
uniform float deltaTime;
uniform uint frameNumber;
uniform uint passIndex;

// Constants
const int MAX_FALL_DISTANCE = 4;  // Maximum blocks to fall at once
const float BASE_MOVEMENT_RATE = 60.0;  // Base rate at 60fps
const float LIQUID_SPREAD_RATE = 100000.0;  // Faster spread for liquids
const int MAX_SPREAD_DISTANCE = 5;  // Maximum horizontal spread distance

// Initialize RNG with better distribution (unchanged)
uint initRNG(ivec3 pos) {
    uint hash = pos.x ^ (pos.y << 8) ^ (pos.z << 16) ^ 
                (frameNumber << 24) ^ (passIndex << 20);
    hash = (hash ^ 61) ^ (hash >> 16);
    hash = hash + (hash << 3);
    hash = hash ^ (hash >> 4);
    hash = hash * 0x27d4eb2d;
    hash = hash ^ (hash >> 15);
    return hash;
}

// Generate random position offset (unchanged)
ivec3 getRandomOffset(uint seed) {
    uint hash = seed;
    hash = (hash ^ 61) ^ (hash >> 16);
    hash = hash * 0x27d4eb2d;
    int x = int(hash % 3u);
    hash = hash ^ (hash >> 16);
    hash = hash * 0x27d4eb2d;
    int y = int(hash % 3u);
    hash = hash ^ (hash >> 16);
    hash = hash * 0x27d4eb2d;
    int z = int(hash % 3u);
    return ivec3(x, y, z);
}

// Check fall distance (unchanged)
int checkFallDistance(ivec3 currentPos) {
    int maxDist = min(MAX_FALL_DISTANCE, currentPos.y);
    for (int dy = 1; dy <= maxDist; dy++) {
        if (atomicCheckVoxel(currentPos + ivec3(0, -dy, 0))) {
            return dy - 1;
        }
    }
    return maxDist;
}

// Check boundary crossing (unchanged)
bool checkCrossBoundaryMove(ivec3 fromPos, ivec3 toPos) {
    if (!isValidPosition(toPos)) {
        return false;
    }
    
    ivec3 chunkPos = toPos / CHUNK_SIZE;
    int chunkIndex = getChunkIndex(chunkPos);
    ivec3 blockInChunk = (toPos / BLOCK_SIZE) & (BLOCK_SIZE - 1);
    int blockIndex = getLocalIndex(blockInChunk);
    
    int chunkArrayIndex = chunkIndex * 4;
    uint chunkArrayPos = chunkArrayIndex + (blockIndex >> 7);
    uint chunkUintIndex = (blockIndex >> 5) & 3;
    uint chunkBitIndex = blockIndex & 31;
    uint chunkBit = 1u << chunkBitIndex;
    
    bool blockExists = (chunkMasks[chunkArrayPos][chunkUintIndex] & chunkBit) != 0;
    
    if (!blockExists) {
        return true;
    }
    
    int globalBlockIndex = chunkIndex * (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) + blockIndex;
    ivec3 voxelInBlock = toPos & (BLOCK_SIZE - 1);
    int voxelIndex = getLocalIndex(voxelInBlock);
    
    int blockArrayIndex = globalBlockIndex * 4;
    uint blockArrayPos = blockArrayIndex + (voxelIndex >> 7);
    uint blockUintIndex = (voxelIndex >> 5) & 3;
    uint blockBitIndex = voxelIndex & 31;
    uint blockBit = 1u << blockBitIndex;
    
    return (blockMasks[blockArrayPos][blockUintIndex] & blockBit) == 0;
}

// Check how far we can spread horizontally in a given direction
int checkSpreadDistance(ivec3 currentPos, ivec3 direction) {
    for (int dist = 1; dist <= MAX_SPREAD_DISTANCE; dist++) {
        ivec3 spreadPos = currentPos + direction * dist;
        
        // Stop if we hit an invalid position or blocked space
        if (!isValidPosition(spreadPos) || 
            !checkCrossBoundaryMove(currentPos, spreadPos) ||
            atomicCheckVoxel(spreadPos)) {
            return dist - 1;
        }
    }
    return MAX_SPREAD_DISTANCE;
}

// New function for horizontal liquid spread
bool tryHorizontalSpread(ivec3 currentPos, inout uint rngState) {
    // Generate random direction order for fair spreading
    int directions[4] = int[4](0, 1, 2, 3);
    
    // Fisher-Yates shuffle for directions
    for (int i = 3; i > 0; i--) {
        int j = int(randomFloat(rngState) * float(i + 1));
        int temp = directions[i];
        directions[i] = directions[j];
        directions[j] = temp;
    }
    
    // Try each direction in random order
    for (int i = 0; i < 4; i++) {
        ivec3 offset;
        switch (directions[i]) {
            case 0: offset = ivec3(1, 0, 0); break;
            case 1: offset = ivec3(-1, 0, 0); break;
            case 2: offset = ivec3(0, 0, 1); break;
            case 3: offset = ivec3(0, 0, -1); break;
        }
        
        // Check maximum spread distance in this direction
        int maxDist = checkSpreadDistance(currentPos, offset);
        if (maxDist > 0) {
            // For more natural movement, we might not always move the full distance
            int actualDist = maxDist;
            if (randomFloat(rngState) < 0.3) { // 30% chance to move shorter distance
                actualDist = 1 + int(randomFloat(rngState) * float(maxDist - 1));
            }
            
            ivec3 spreadPos = currentPos + offset * actualDist;
            if (atomicMoveVoxel(currentPos, spreadPos)) {
                return true;
            }
        }
    }
    
    return false;
}

// Modified tryMove to handle both powder and liquid
bool tryMove(ivec3 currentPos, bool isLiquid, inout uint rngState) {
    float movementChance = min(1.0, deltaTime * (isLiquid ? LIQUID_SPREAD_RATE : BASE_MOVEMENT_RATE));
    if (randomFloat(rngState) > movementChance) {
        return false;
    }
    
    // Check vertical movement first
    int fallDist = checkFallDistance(currentPos);
    if (fallDist > 0) {
        ivec3 downPos = currentPos + ivec3(0, -fallDist, 0);
        if (checkCrossBoundaryMove(currentPos, downPos)) {
            if (atomicMoveVoxel(currentPos, downPos)) {
                return true;
            }
        }
    }
    
    // For liquids, try horizontal spread even if we can't fall
    if (isLiquid) {
        if (tryHorizontalSpread(currentPos, rngState)) {
            return true;
        }
    }
    
    // Try diagonal moves for both liquids and powders
    int xDir = randomFloat(rngState) < 0.5 ? -1 : 1;
    int zDir = randomFloat(rngState) < 0.5 ? -1 : 1;
    
    for (int attempt = 0; attempt < 4; attempt++) {
        ivec3 diagPos = currentPos + ivec3(
            attempt < 2 ? xDir : 0,
            -1,
            attempt < 2 ? 0 : zDir
        );
        
        if (checkCrossBoundaryMove(currentPos, diagPos)) {
            if (atomicMoveVoxel(currentPos, diagPos)) {
                return true;
            }
        }
        
        if (attempt == 1) {
            xDir = -xDir;
        } else if (attempt == 3) {
            zDir = -zDir;
        }
    }
    
    return false;
}

void main() {
    ivec3 basePos = ivec3(gl_GlobalInvocationID.xyz);
    uint seed = basePos.x ^ (basePos.y << 8) ^ (basePos.z << 16) ^ 
                (frameNumber << 24) ^ (passIndex << 20);
    ivec3 offset = getRandomOffset(seed);
    ivec3 pos = basePos * 3 + offset;
    
    if (!isValidPosition(pos)) {
        return;
    }
    
    if (!atomicCheckVoxel(pos)) {
        return;
    }

    ivec3 chunkPos = pos / CHUNK_SIZE;
    int chunkIndex = getChunkIndex(chunkPos);
    ivec3 blockInChunk = (pos / BLOCK_SIZE) % BLOCK_SIZE;
    int blockLocalIndex = getLocalIndex(blockInChunk);
    ivec3 voxelInBlock = pos % BLOCK_SIZE;
    int voxelLocalIndex = getLocalIndex(voxelInBlock);
    
    int globalBlockIndex = globalIndex(chunkIndex, blockLocalIndex);
    int globalVoxelIndex = globalIndex(globalBlockIndex, voxelLocalIndex);
    
    uint materialId = getMaterial(globalVoxelIndex);
    Material material = materials[materialId];
    
    // Skip if neither powder nor liquid
    if (!material.isPowder && !material.isLiquid) {
        return;
    }
    
    uint rngState = initRNG(pos);
    tryMove(pos, material.isLiquid, rngState);
    
    memoryBarrier();
}