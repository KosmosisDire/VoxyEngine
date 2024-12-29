#version 460

#include "common.glsl"
#include "common-move.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// Uniforms
uniform float deltaTime;
uniform uint frameNumber;
uniform ivec3 minCoord;
uniform ivec3 maxCoord;
uniform float currentScale;
uniform ivec3 cameraPos;
uniform vec3 cameraDir;

// Constants
const int MAX_FALL_DISTANCE = 5;
const int MAX_SPREAD_DISTANCE = 15;
const int DIAGONAL_DIRECTIONS = 8;
const float LIQUID_EVAPORATION_CHANCE = 0.001;
const float LIQUID_FILL_CHANCE = 0.006;  // Matched with evaporation for equilibrium

const ivec3 DIAGONAL_OFFSETS[8] = ivec3[8](
    ivec3(1, 0, 1),   // Northeast
    ivec3(1, 0, -1),  // Southeast
    ivec3(-1, 0, 1),  // Northwest
    ivec3(-1, 0, -1), // Southwest
    ivec3(1, 0, 0),   // East
    ivec3(-1, 0, 0),  // West
    ivec3(0, 0, 1),   // North
    ivec3(0, 0, -1)   // South
);

const ivec3 HORIZONTAL_DIRECTIONS[4] = ivec3[4](
    ivec3(1, 0, 0),   // East
    ivec3(-1, 0, 0),  // West
    ivec3(0, 0, 1),   // North
    ivec3(0, 0, -1)   // South
);

// Helper function to get material ID at a position
uint getMaterialAtPosition(ivec3 pos) {
    ivec3 chunkPos = pos / CHUNK_SIZE;
    int chunkIndex = getChunkIndex(chunkPos);
    ivec3 blockInChunk = (pos / BLOCK_SIZE) % BLOCK_SIZE;
    int blockLocalIndex = getLocalIndex(blockInChunk);
    ivec3 voxelInBlock = pos % BLOCK_SIZE;
    int voxelLocalIndex = getLocalIndex(voxelInBlock);
    
    int globalBlockIndex = globalIndex(chunkIndex, blockLocalIndex);
    int globalVoxelIndex = globalIndex(globalBlockIndex, voxelLocalIndex);
    
    return getMaterial(globalVoxelIndex);
}

// Helper function to add a voxel with a specific material
bool atomicAddVoxel(ivec3 pos, uint materialId) {
    if (!isValidPosition(pos) || atomicCheckVoxel(pos)) {
        return false;
    }

    ivec3 chunkPos = pos / CHUNK_SIZE;
    int chunkIndex = getChunkIndex(chunkPos);
    ivec3 blockInChunk = (pos / BLOCK_SIZE) & (BLOCK_SIZE - 1);
    int blockIndex = getLocalIndex(blockInChunk);
    ivec3 voxelInBlock = pos & (BLOCK_SIZE - 1);
    int voxelIndex = getLocalIndex(voxelInBlock);
    
    int globalBlockIndex = chunkIndex * (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) + blockIndex;
    int globalVoxelIndex = globalIndex(globalBlockIndex, voxelIndex);
    
    // Set block mask bit
    int blockArrayIndex = globalBlockIndex * 4;
    uint blockArrayPos = blockArrayIndex + (voxelIndex >> 7);
    uint blockUintIndex = (voxelIndex >> 5) & 3;
    uint blockBitIndex = voxelIndex & 31;
    
    atomicOr(blockMasks[blockArrayPos][blockUintIndex], (1u << blockBitIndex));
    
    // Set chunk mask bit
    int chunkArrayIndex = chunkIndex * 4;
    uint chunkArrayPos = chunkArrayIndex + (blockIndex >> 7);
    uint chunkUintIndex = (blockIndex >> 5) & 3;
    uint chunkBitIndex = blockIndex & 31;
    
    atomicOr(chunkMasks[chunkArrayPos][chunkUintIndex], (1u << chunkBitIndex));
    
    // Set material ID
    setMaterial(globalVoxelIndex, materialId);
    
    return true;
}

// Simplified RNG
uint initRNG(ivec3 pos) {
    uint hash = pos.x ^ (pos.y << 8) ^ (pos.z << 16) ^ (frameNumber << 24);
    hash = (hash ^ 61) ^ (hash >> 16);
    hash = hash * 0x27d4eb2d;
    return hash ^ (hash >> 15);
}

// Movement validity checks
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
    
    if ((chunkMasks[chunkArrayPos][chunkUintIndex] & (1u << chunkBitIndex)) == 0) {
        return true;
    }
    
    int globalBlockIndex = chunkIndex * (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) + blockIndex;
    ivec3 voxelInBlock = toPos & (BLOCK_SIZE - 1);
    int voxelIndex = getLocalIndex(voxelInBlock);
    
    int blockArrayIndex = globalBlockIndex * 4;
    uint blockArrayPos = blockArrayIndex + (voxelIndex >> 7);
    uint blockUintIndex = (voxelIndex >> 5) & 3;
    uint blockBitIndex = voxelIndex & 31;
    
    return (blockMasks[blockArrayPos][blockUintIndex] & (1u << blockBitIndex)) == 0;
}

// Vertical movement check
int checkFallDistance(ivec3 currentPos) {
    int maxDist = min(MAX_FALL_DISTANCE, currentPos.y - minCoord.y);
    for (int dy = 1; dy <= maxDist; dy++) {
        if (atomicCheckVoxel(currentPos + ivec3(0, -dy, 0))) {
            return dy - 1;
        }
    }
    return maxDist;
}

// Enhanced powder movement with 8-directional support
bool tryMovePowder(ivec3 currentPos, inout uint rngState) {
    // Try vertical fall first
    int fallDist = checkFallDistance(currentPos);
    if (fallDist > 0) {
        ivec3 downPos = currentPos - ivec3(0, fallDist, 0);
        if (checkCrossBoundaryMove(currentPos, downPos) && atomicMoveVoxel(currentPos, downPos)) {
            return true;
        }
    }
    
    // Shuffle diagonal directions for random movement
    int directions[DIAGONAL_DIRECTIONS];
    for(int i = 0; i < DIAGONAL_DIRECTIONS; i++) {
        directions[i] = i;
    }
    
    // Fisher-Yates shuffle
    for(int i = DIAGONAL_DIRECTIONS - 1; i > 0; i--) {
        int j = int(randomFloat(rngState) * float(i + 1));
        int temp = directions[i];
        directions[i] = directions[j];
        directions[j] = temp;
    }
    
    // Try diagonal movements
    for(int i = 0; i < DIAGONAL_DIRECTIONS; i++) {
        ivec3 offset = DIAGONAL_OFFSETS[directions[i]];
        ivec3 diagPos = currentPos + offset - ivec3(0, 1, 0); // Diagonal down movement
        
        // Check if the path is clear
        if (checkCrossBoundaryMove(currentPos, diagPos) && 
            !atomicCheckVoxel(diagPos) &&
            atomicMoveVoxel(currentPos, diagPos)) {
            return true;
        }
    }
    
    return false;
}

// Check if a position is valid for liquid fill
float fillChance(ivec3 pos, uint sourceMaterialId) {
    // Must be empty
    if (atomicCheckVoxel(pos)) {
        return 0.0;
    }
    
    // Must be empty above
    if (atomicCheckVoxel(pos + ivec3(0, 1, 0))) {
        return 0.0;
    }
    
    // Check all horizontal directions have matching liquid
    int surroundCount = 0;
    for (int i = 0; i < 4; i++) {
        ivec3 checkPos = pos + HORIZONTAL_DIRECTIONS[i];
        if (atomicCheckVoxel(checkPos) && getMaterialAtPosition(checkPos) == sourceMaterialId) {
            surroundCount++;
        }
    }
    
    return surroundCount >= 2 ? surroundCount / 4.0 : 0.0;
}

// Check if a position is valid for evaporation
float evaporationChance(ivec3 pos) {
    // Check for block underneath (must exist)
    if (!atomicCheckVoxel(pos + ivec3(0, -1, 0))) {
        return 0.0;
    }
    
    // Check for block above (must not exist)
    if (atomicCheckVoxel(pos + ivec3(0, 1, 0))) {
        return 0.0;
    }

    // Check if not surrounded horizontally (must not be surrounded)
    int surroundCount = 0;
    for (int i = 0; i < 4; i++) {
        ivec3 checkPos = pos + HORIZONTAL_DIRECTIONS[i];
        if (atomicCheckVoxel(checkPos)) {
            surroundCount++;
        }
    }

    return surroundCount < 4 ? (4 - surroundCount) / 4.0 : 0.0;
}

// Liquid spreading behavior
bool tryMoveLiquid(ivec3 currentPos, inout uint rngState)
{
    ivec3 newPos = currentPos;
    // Try falling first
    int fallDist = checkFallDistance(newPos);
    if (fallDist > 0) {
        ivec3 downPos = newPos - ivec3(0, fallDist, 0);
        newPos = downPos;
    }
    
    // Check horizontal spread
    bool moved = false;

    int randomStart = int(randomFloat(rngState) * 8);

    for(int i = 0; i < 8; i++) {
        ivec3 dir = DIAGONAL_OFFSETS[(randomStart + i) % 8];
        int lastFailedDist = 1;
        for(int dist = 1; dist <= MAX_SPREAD_DISTANCE; dist++) {
            ivec3 spreadPos = ivec3(newPos + dir * dist * dist);
            
            if (!checkCrossBoundaryMove(newPos, spreadPos) || atomicCheckVoxel(spreadPos)) {
                break;
            }

            // if we can move down from here then stop
            if (checkFallDistance(spreadPos) > 0 || randomFloat(rngState) < (1.0 / MAX_SPREAD_DISTANCE) / 5.0)
            {
                moved = true;
                newPos += ivec3(dir * dist);
                break;
            }

            lastFailedDist = dist;
        }
        if (moved) break;
    }

    // Move to the final position
    if(checkCrossBoundaryMove(currentPos, newPos) && atomicMoveVoxel(currentPos, newPos)) {
        moved = true;
    }
    
    return moved;
}


void main()
{
    for (int i = 0; i < 5; i++)
    {
        ivec3 pos = ivec3(gl_GlobalInvocationID.xyz) + minCoord;
        
        if (any(lessThan(pos, minCoord)) || any(greaterThan(pos, maxCoord))) {
            return;
        }

        if (!atomicCheckVoxel(pos)) {
            return;
        }

        uint rngState = initRNG(pos);
        
        // Get material information
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
        
        bool moved = false;
        // Process movement based on material type
        if (material.isPowder) {
            moved = tryMovePowder(pos, rngState);
        } else if (material.isLiquid) {
            moved = tryMoveLiquid(pos, rngState);
            
            if (!moved)
            {
                ivec3 sidePos = pos + ivec3(-1, 0, 0);
                if (randomFloat(rngState) < fillChance(sidePos, materialId) * LIQUID_FILL_CHANCE * 0.25) {
                    atomicAddVoxel(sidePos, materialId);
                }

                sidePos = pos + ivec3(1, 0, 0);
                if (randomFloat(rngState) < fillChance(sidePos, materialId) * LIQUID_FILL_CHANCE * 0.25) {
                    atomicAddVoxel(sidePos, materialId);
                }

                sidePos = pos + ivec3(0, 0, -1);
                if (randomFloat(rngState) < fillChance(sidePos, materialId) * LIQUID_FILL_CHANCE * 0.25) {
                    atomicAddVoxel(sidePos, materialId);
                }

                sidePos = pos + ivec3(0, 0, 1);
                if (randomFloat(rngState) < fillChance(sidePos, materialId) * LIQUID_FILL_CHANCE * 0.25) {
                    atomicAddVoxel(sidePos, materialId);
                }

                if (randomFloat(rngState) < evaporationChance(pos) * LIQUID_EVAPORATION_CHANCE) {
                    atomicClearVoxel(pos);
                }
            }
        }
    }
}