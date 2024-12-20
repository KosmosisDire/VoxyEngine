#version 460

#include "common.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

uniform vec3 cornerPos;
uniform vec3 cubeSize;
uniform int materialId;
uniform bool clear;

// Buffer to store clear results (what types of voxels were broken and how many of each)
struct ClearResult {
    uint materialId;
    uint count;
};

layout(std430, binding = 0) buffer ClearResultsBuffer {
    ClearResult clearResults[]; // max is 255 different materials so same is the buffer size
};

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);
    
    // Check if we're within the desired cube size
    if (any(greaterThanEqual(pos, ivec3(cubeSize)))) {
        return;
    }
    
    // Convert local position to world position
    ivec3 worldPos = ivec3(cornerPos) + pos;
    
    // Get chunk coordinates
    ivec3 chunkPos = worldPos / CHUNK_SIZE;
    int chunkIndex = getChunkIndex(chunkPos);
    
    // Get block coordinates within chunk
    ivec3 blockInChunk = (worldPos / BLOCK_SIZE) % BLOCK_SIZE;
    int blockIndex = getLocalIndex(blockInChunk);
    
    // Get voxel coordinates within block
    ivec3 voxelInBlock = worldPos % BLOCK_SIZE;
    int voxelIndex = getLocalIndex(voxelInBlock);
    
    if (clear) {
        // Check if there's a voxel to clear
        Bitmask chunkMask = getChunkBitmask(chunkIndex);
        if (!getBit(chunkMask, blockIndex)) {
            return;
        }
        
        int globalBlockIndex = globalIndex(chunkIndex, blockIndex);
        Bitmask blockMask = getBlockBitmask(globalBlockIndex);
        if (!getBit(blockMask, voxelIndex)) {
            return;
        }
        
        // Get material before clearing
        uint oldMaterial = getMaterial(globalIndex(globalBlockIndex, voxelIndex));
        
        // Atomically increment the count for this material type
        // Note: This is a simple linear search through the buffer
        // Could be optimized with a more sophisticated approach if needed
        for (int i = 0; i < 255; i++) {
            if (clearResults[i].materialId == oldMaterial || clearResults[i].count == 0) {
                if (clearResults[i].count == 0) {
                    clearResults[i].materialId = oldMaterial;
                }
                atomicAdd(clearResults[i].count, 1);
                break;
            }
        }
        
        // Clear the voxel
        atomicAnd(blockMasks[globalBlockIndex * 4 + (voxelIndex >> 7)][voxelIndex >> 5 & 3], 
                 ~(1u << (voxelIndex & 31)));
    } else {
        // Setting a voxel
        // First set the chunk bit
        atomicSetChunkBit(chunkIndex, blockIndex);
        
        // Then set the block bit
        int globalBlockIndex = globalIndex(chunkIndex, blockIndex);
        atomicSetBlockBit(globalBlockIndex, voxelIndex);
        
        // Finally set the material
        setMaterial(globalIndex(globalBlockIndex, voxelIndex), uint(materialId));
    }
}