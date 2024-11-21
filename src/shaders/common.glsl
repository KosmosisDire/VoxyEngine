#version 460

// Constants
const float EPSILON = 0.001;
const int MAX_STEPS = 256;
const float MAX_DIST = 1000.0;
const int SPLIT_SIZE = 5;
const int CHUNK_SIZE = SPLIT_SIZE * SPLIT_SIZE; // 16 voxels per chunk dimension
const float CHUNK_SIZE_INV = 1.0 / CHUNK_SIZE;
const float SPLIT_SIZE_INV = 1.0 / SPLIT_SIZE;
const int GRID_SIZE = 128; // 16x16x16 grid of chunks
const uint INVALID_INDEX = 4294967295u; 

// one bitmask per chunk representing whether each outer block is subdivided
layout(std430, binding = 0) buffer ChunkBitmasksLow {
    uvec4 chunkMasksLow[]; // 125 bits for 125 blocks per chunk, extra bits are flags (undecided)
};
// 64 voxel bitmasks per chunk
layout(std430, binding = 1) buffer VoxelBitmasksLow {
    uvec4 voxelMasksLow[]; // 125 bits for 125 voxels per block, extra bits are flags (for non homogenous for example)
};
layout(std430, binding = 2) buffer BlockData {
    uint blockData[]; // 1 byte per block just a pointer to the material index
};

// not used for now, leaving here until I implement voxel data
layout(std430, binding = 3) buffer VoxelData {
    uint voxelData[]; // 1 byte per voxel for nonhomogenous blocks
};

// materials 256 total
struct Material {
    vec3 color;
    bool emissive;
};
layout(std430, binding = 4) buffer Materials {
    Material materials[];
};

// Utility functions
bool getBit(int index, uint maskLow, uint maskHigh) {
    if (index < 32) {
        return (maskLow & (1u << uint(index))) != 0u;
    } else {
        return (maskHigh & (1u << uint(index - 32))) != 0u;
    }
}

void setBit(inout uint maskLow, inout uint maskHigh, int index) {
    if (index < 32) {
        maskLow |= (1u << uint(index));
    } else {
        maskHigh |= (1u << uint(index - 32));
    }
}

int getSplitIndexLocal(ivec3 coord)
{
    return coord.x + coord.z * SPLIT_SIZE + coord.y * SPLIT_SIZE * SPLIT_SIZE;
}

int getChunkIndex(ivec3 gridPos)
{
    return gridPos.x + gridPos.z * GRID_SIZE + gridPos.y * GRID_SIZE * GRID_SIZE;
}

ivec3 getChunkGridPos(int chunkIndex) {
    int x = chunkIndex % GRID_SIZE;
    int z = (chunkIndex / GRID_SIZE) % GRID_SIZE;
    int y = chunkIndex / (GRID_SIZE * GRID_SIZE);
    return ivec3(x, y, z);
}


// Color packing functions
vec3 unpackColor(uint bits) {
    // Extract 5 bits per channel from the first 15 bits
    float r = float(bits & 31u) / 31.0;
    float g = float((bits >> 5u) & 31u) / 31.0;
    float b = float((bits >> 10u) & 31u) / 31.0;
    return vec3(r, g, b);
}

uint packColor(vec3 color) {
    // Pack into 5 bits per channel
    uint r = uint(clamp(color.r * 31.0, 0.0, 31.0));
    uint g = uint(clamp(color.g * 31.0, 0.0, 31.0));
    uint b = uint(clamp(color.b * 31.0, 0.0, 31.0));
    return r | (g << 5u) | (b << 10u);
}

bool getEmissive(uint bits) {
    return (bits & (1u << 15u)) != 0u;
}

uint setEmissive(uint bits, bool emissive) {
    if (emissive) {
        return bits | (1u << 15u);
    } else {
        return bits & ~(1u << 15u);
    }
}

uint getLightStrength(uint bits) {
    return (bits >> 16u) & 255u;
}

uint setLightStrength(uint bits, uint strength) {
    return (bits & 0x0000FFFFu) | (strength << 16u);
}

uint getAveragingCounter(uint bits) {
    return (bits >> 24u) & 255u;
}

uint setAveragingCounter(uint bits, uint counter) {
    return (bits & 0x00FFFFFFu) | (counter << 24u);
}

bool intersectBox(vec3 origin, vec3 invDir, vec3 boxMin, vec3 boxMax, out float tMin, out float tMax) {
    vec3 t0 = (boxMin - origin) * invDir;
    vec3 t1 = (boxMax - origin) * invDir;
    vec3 tMinVec = min(t0, t1);
    vec3 tMaxVec = max(t0, t1);
    tMin = max(max(tMinVec.x, tMinVec.y), tMinVec.z);
    tMax = min(min(tMaxVec.x, tMaxVec.y), tMaxVec.z);
    return tMin <= tMax && tMax > 0.0;
}

void setVoxelColor(uint index, vec3 color, bool emissive) {
    uint packedColor = packColor(color);
    uint packedData = setEmissive(packedColor, emissive);
    voxelData[index] = packedData;
}

// Allocates a new block and returns its index in voxelData
uint allocateVoxelBlock(int chunkIndex, int blockLocalIndex) {
    int blockIndexOffset = chunkIndex * (SPLIT_SIZE * SPLIT_SIZE * SPLIT_SIZE) + blockLocalIndex;
    
    // Check for existing block first
    uint existingIndex = blockIndices[blockIndexOffset];
    if (existingIndex != INVALID_INDEX) {
        return existingIndex;
    }
    
    // First try to claim the block position with a temporary value
    uint result = atomicCompSwap(blockIndices[blockIndexOffset], INVALID_INDEX, INVALID_INDEX - 1);
    if (result != INVALID_INDEX) {
        // Another thread got here first
        return result;
    }
    
    // We successfully claimed the block position, now get a new index
    uint newIndex = dequeueBlockIndex();
    if (newIndex == INVALID_INDEX) {
        // Failed to get new index, restore the block position
        atomicExchange(blockIndices[blockIndexOffset], INVALID_INDEX);
        return INVALID_INDEX;
    }
    
    // Set the actual index
    atomicExchange(blockIndices[blockIndexOffset], newIndex);
    
    // Initialize the new block
    atomicAnd(voxelMasksLow[blockIndexOffset], 0u);
    atomicAnd(voxelMasksHigh[blockIndexOffset], 0u);
    
    // Set chunk bitmask
    if (blockLocalIndex < 32) {
        atomicOr(chunkMasksLow[chunkIndex], 1u << uint(blockLocalIndex));
    } else {
        atomicOr(chunkMasksHigh[chunkIndex], 1u << uint(blockLocalIndex - 32));
    }
    
    return newIndex;
}

// Frees a block and returns it to the free queue
void freeVoxelBlock(int chunkIndex, int blockLocalIndex)
{
    // Clear the chunk bitmask to indicate this block is free
    if (blockLocalIndex < 32) {
        atomicAnd(chunkMasksLow[chunkIndex], ~(1u << uint(blockLocalIndex)));
    } else {
        atomicAnd(chunkMasksHigh[chunkIndex], ~(1u << uint(blockLocalIndex - 32)));
    }

    // Clear the block index
    int blockIndexOffset = chunkIndex * (SPLIT_SIZE * SPLIT_SIZE * SPLIT_SIZE) + blockLocalIndex;
    blockIndices[blockIndexOffset] = INVALID_INDEX;

    // Clear the voxel bitmasks for this block
    voxelMasksLow[blockIndexOffset] = 0u;
    voxelMasksHigh[blockIndexOffset] = 0u;

    // Add the block to the free queue
    enqueueBlockIndex(blockIndexOffset);
}

// Sets a voxel in world space
void setVoxel(int chunkIndex, int blockLocalIndex, int voxelLocalIndex, vec3 color, bool emissive) {
    // Convert world position to grid coordinates
    int blockIndexOffset = chunkIndex * SPLIT_SIZE * SPLIT_SIZE * SPLIT_SIZE + blockLocalIndex;

    // Get or allocate block
    uint blockDataIndex = allocateVoxelBlock(chunkIndex, blockLocalIndex);
    if (blockDataIndex == INVALID_INDEX) {
        return;
    }

    // Set the voxel bitmask - do this BEFORE writing data to ensure consistency
    if (voxelLocalIndex < 32) {
        atomicOr(voxelMasksLow[blockIndexOffset], 1u << uint(voxelLocalIndex));
    } else {
        atomicOr(voxelMasksHigh[blockIndexOffset], 1u << uint(voxelLocalIndex - 32));
    }

    // Set the voxel data
    uint voxelIndex = uint(blockDataIndex + voxelLocalIndex);
    uint packedColor = packColor(color);
    uint packedData = setEmissive(packedColor, emissive);
    atomicExchange(voxelData[voxelIndex], packedData);
}

// Clears a voxel in world space
void clearVoxel(vec3 worldPos)
{
    // Convert world position to grid coordinates
    ivec3 gridPos = ivec3(floor(worldPos));
    int chunkIndex = getChunkIndex(ivec3(gridPos * CHUNK_SIZE_INV));
    int blockIndexLocal = getSplitIndexLocal(ivec3((gridPos / SPLIT_SIZE) % SPLIT_SIZE));
    int blockIndex = chunkIndex * SPLIT_SIZE * SPLIT_SIZE * SPLIT_SIZE + blockIndexLocal;
    int voxelIndexLocal = getSplitIndexLocal(gridPos % SPLIT_SIZE);

    uint maskLow = voxelMasksLow[blockIndex];
    uint maskHigh = voxelMasksHigh[blockIndex];

    // Check if the voxel is already set
    if (!getBit(voxelIndexLocal, maskLow, maskHigh))
    {
        // Voxel is already clear
        return;
    }

    // Clear the voxel bitmask atomically
    if (voxelIndexLocal < 32) {
        atomicAnd(voxelMasksLow[blockIndex], ~(1u << uint(voxelIndexLocal)));
    } else {
        atomicAnd(voxelMasksHigh[blockIndex], ~(1u << uint(voxelIndexLocal - 32)));
    }

    // check if the block is empty
    if (voxelMasksLow[blockIndex] == 0u && voxelMasksHigh[blockIndex] == 0u)
    {
        // free the block
        freeVoxelBlock(chunkIndex, blockIndexLocal);
    }
}

// PCG Random number generator
uint pcg(inout uint state)
{
    uint prev = state * 747796405u + 2891336453u;
    uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    state = prev;
    return (word >> 22u) ^ word;
}

float randomFloat(inout uint seed) 
{
    return float(pcg(seed)) * (1.0 / 4294967296.0);
}