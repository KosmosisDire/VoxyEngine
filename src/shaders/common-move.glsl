// Constants for bit manipulation
const uint BLOCK_BITS = 5;  // 32 bits per uint
const uint BLOCK_MASK = 31;  // 0b11111
const uint ARRAY_BITS = 7;   // 128 bits per array index
const uint ARRAY_MASK = 127; // 0b1111111

// Pre-calculate grid bounds for faster checks
const ivec3 GRID_BOUNDS = ivec3(CHUNK_SIZE * GRID_SIZE);

// Optimized position validity check
bool isValidPosition(ivec3 pos) {
    return all(greaterThanEqual(pos, ivec3(0))) && 
           all(lessThan(pos, GRID_BOUNDS));
}

// Calculate array indices for a position
struct BitIndices {
    uint arrayIndex;
    uint uintIndex;
    uint bitIndex;
};

BitIndices calculateBitIndices(int baseIndex, int localIndex) {
    BitIndices indices;
    indices.arrayIndex = baseIndex + (localIndex >> ARRAY_BITS);
    indices.uintIndex = (localIndex >> BLOCK_BITS) & 3;
    indices.bitIndex = localIndex & BLOCK_MASK;
    return indices;
}

// Optimized voxel check
bool atomicCheckVoxel(ivec3 pos) {
    if (!isValidPosition(pos)) {
        return false;
    }
    
    // Calculate chunk and block positions in one step
    ivec3 chunkPos = pos / CHUNK_SIZE;
    int chunkIndex = getChunkIndex(chunkPos);
    
    // Calculate block indices
    ivec3 blockInChunk = (pos / BLOCK_SIZE) & (BLOCK_SIZE - 1);
    int blockIndex = getLocalIndex(blockInChunk);
    
    // Get chunk mask indices
    BitIndices chunkIndices = calculateBitIndices(chunkIndex * 4, blockIndex);
    
    // Early exit if chunk bit is not set
    uint chunkBit = 1u << chunkIndices.bitIndex;
    if ((chunkMasks[chunkIndices.arrayIndex][chunkIndices.uintIndex] & chunkBit) == 0) {
        return false;
    }
    
    // Calculate global block index
    int globalBlockIndex = chunkIndex * (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) + blockIndex;
    
    // Calculate voxel indices
    ivec3 voxelInBlock = pos & (BLOCK_SIZE - 1);
    int voxelIndex = getLocalIndex(voxelInBlock);
    
    // Get block mask indices
    BitIndices blockIndices = calculateBitIndices(globalBlockIndex * 4, voxelIndex);
    
    // Check block bit
    uint blockBit = 1u << blockIndices.bitIndex;
    return (blockMasks[blockIndices.arrayIndex][blockIndices.uintIndex] & blockBit) != 0;
}

int getVoxelMaterial(ivec3 pos) {
    // Check if position is valid
    if (!isValidPosition(pos)) {
        return -1;
    }
    
    // First check if the voxel is occupied
    if (!atomicCheckVoxel(pos)) {
        return -1;
    }
    
    // Calculate indices to get material
    ivec3 chunkPos = pos / CHUNK_SIZE;
    int chunkIndex = getChunkIndex(chunkPos);
    ivec3 blockInChunk = (pos / BLOCK_SIZE) & (BLOCK_SIZE - 1);
    int blockIndex = getLocalIndex(blockInChunk);
    int globalBlockIndex = chunkIndex * (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) + blockIndex;
    ivec3 voxelInBlock = pos & (BLOCK_SIZE - 1);
    int voxelIndex = getLocalIndex(voxelInBlock);
    
    // Get the global index for material lookup
    int globalVoxelIndex = globalIndex(globalBlockIndex, voxelIndex);
    
    // Return the material
    return int(getMaterial(globalVoxelIndex));
}

// Optimized voxel claim
bool atomicClaimVoxel(ivec3 pos) {
    if (!isValidPosition(pos)) {
        return false;
    }
    
    // Calculate all indices in one pass
    ivec3 chunkPos = pos / CHUNK_SIZE;
    int chunkIndex = getChunkIndex(chunkPos);
    ivec3 blockInChunk = (pos / BLOCK_SIZE) & (BLOCK_SIZE - 1);
    int blockIndex = getLocalIndex(blockInChunk);
    
    // Set chunk bit
    BitIndices chunkIndices = calculateBitIndices(chunkIndex * 4, blockIndex);
    uint chunkBit = 1u << chunkIndices.bitIndex;
    atomicOr(chunkMasks[chunkIndices.arrayIndex][chunkIndices.uintIndex], chunkBit);
    
    // Calculate block indices
    int globalBlockIndex = chunkIndex * (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) + blockIndex;
    ivec3 voxelInBlock = pos & (BLOCK_SIZE - 1);
    int voxelIndex = getLocalIndex(voxelInBlock);
    
    // Set block bit
    BitIndices blockIndices = calculateBitIndices(globalBlockIndex * 4, voxelIndex);
    uint blockBit = 1u << blockIndices.bitIndex;
    
    // Return true if bit was not previously set
    return (atomicOr(blockMasks[blockIndices.arrayIndex][blockIndices.uintIndex], blockBit) & blockBit) == 0;
}

// Optimized voxel clear
void atomicClearVoxel(ivec3 pos) {
    if (!isValidPosition(pos)) {
        return;
    }
    
    // Calculate all indices at once
    ivec3 chunkPos = pos / CHUNK_SIZE;
    int chunkIndex = getChunkIndex(chunkPos);
    ivec3 blockInChunk = (pos / BLOCK_SIZE) & (BLOCK_SIZE - 1);
    int blockIndex = getLocalIndex(blockInChunk);
    int globalBlockIndex = chunkIndex * (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) + blockIndex;
    ivec3 voxelInBlock = pos & (BLOCK_SIZE - 1);
    int voxelIndex = getLocalIndex(voxelInBlock);
    
    // Clear block bit
    BitIndices blockIndices = calculateBitIndices(globalBlockIndex * 4, voxelIndex);
    uint blockBit = ~(1u << blockIndices.bitIndex);
    atomicAnd(blockMasks[blockIndices.arrayIndex][blockIndices.uintIndex], blockBit);
}

void atomicSetVoxel(ivec3 pos, uint materialId) {
    if (!isValidPosition(pos)) {
        return;
    }
    
    // Calculate all indices at once
    ivec3 chunkPos = pos / CHUNK_SIZE;
    int chunkIndex = getChunkIndex(chunkPos);
    ivec3 blockInChunk = (pos / BLOCK_SIZE) & (BLOCK_SIZE - 1);
    int blockIndex = getLocalIndex(blockInChunk);
    int globalBlockIndex = chunkIndex * (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) + blockIndex;
    ivec3 voxelInBlock = pos & (BLOCK_SIZE - 1);
    int voxelIndex = getLocalIndex(voxelInBlock);
    
    // Set material
    setMaterial(globalIndex(globalBlockIndex, voxelIndex), materialId);

    // Set block bit
    BitIndices blockIndices = calculateBitIndices(globalBlockIndex * 4, voxelIndex);
    uint blockBit = 1u << blockIndices.bitIndex;
    atomicOr(blockMasks[blockIndices.arrayIndex][blockIndices.uintIndex], blockBit);

    // Set chunk bit
    BitIndices chunkIndices = calculateBitIndices(chunkIndex * 4, blockIndex);
    uint chunkBit = 1u << chunkIndices.bitIndex;
    atomicOr(chunkMasks[chunkIndices.arrayIndex][chunkIndices.uintIndex], chunkBit);
}

bool atomicMoveVoxel(ivec3 fromPos, ivec3 toPos) {
    if (!isValidPosition(toPos)) {
        return false;
    }
    
    // Get source material before moving
    ivec3 fromChunkPos = fromPos / CHUNK_SIZE;
    int fromChunkIndex = getChunkIndex(fromChunkPos);
    ivec3 fromBlockInChunk = (fromPos / BLOCK_SIZE) % BLOCK_SIZE;
    int fromBlockIndex = getLocalIndex(fromBlockInChunk);
    int fromGlobalBlockIndex = fromChunkIndex * (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) + fromBlockIndex;
    ivec3 fromVoxelInBlock = fromPos % BLOCK_SIZE;
    int fromVoxelIndex = getLocalIndex(fromVoxelInBlock);
    int fromGlobalVoxelIndex = globalIndex(fromGlobalBlockIndex, fromVoxelIndex);
    uint materialId = getMaterial(fromGlobalVoxelIndex);
    
    // Try to claim destination first
    if (!atomicClaimVoxel(toPos)) {
        return false;
    }
    
    // Verify source exists
    if (!atomicCheckVoxel(fromPos)) {
        // If source doesn't exist, undo the claim
        atomicClearVoxel(toPos);
        return false;
    }
    
    // Set destination material before clearing source
    ivec3 toChunkPos = toPos / CHUNK_SIZE;
    int toChunkIndex = getChunkIndex(toChunkPos);
    ivec3 toBlockInChunk = (toPos / BLOCK_SIZE) % BLOCK_SIZE;
    int toBlockIndex = getLocalIndex(toBlockInChunk);
    int toGlobalBlockIndex = toChunkIndex * (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) + toBlockIndex;
    ivec3 toVoxelInBlock = toPos % BLOCK_SIZE;
    int toVoxelIndex = getLocalIndex(toVoxelInBlock);
    int toGlobalVoxelIndex = globalIndex(toGlobalBlockIndex, toVoxelIndex);
    setMaterial(toGlobalVoxelIndex, materialId);
    
    // Clear the source material and occupancy
    setMaterial(fromGlobalVoxelIndex, 0); // Set source to empty material
    atomicClearVoxel(fromPos);
    
    return true;
}