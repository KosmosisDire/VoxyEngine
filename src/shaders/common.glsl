#version 460

// Core Constants
const float EPSILON = 0.001;
const int MAX_STEPS = 256;
const float MAX_DIST = 1000.0;

// Structure Constants
const int BLOCK_SIZE = 5;            // 5x5x5 blocks per chunk
const int CHUNK_SIZE = BLOCK_SIZE * BLOCK_SIZE;
const int GRID_SIZE = 128;           // 128x128x128 chunks
const float VOXEL_SIZE = 1.0 / CHUNK_SIZE;
const uint INVALID_INDEX = 4294967295u;

// Material System
struct Material {
    vec3 color;
    float metallic;
    float roughness;
    float emission;
    float noiseSize;
    float noiseStrength;
};

// Buffer Bindings
layout(std430, binding = 0) buffer ChunkMasks {
    uvec4 chunkMasks[];    // 125 bits per chunk for block occupancy (5x5x5 blocks)
};

layout(std430, binding = 1) buffer BlockMasks {
    uvec4 blockMasks[];    // 125 bits per block for voxel occupancy (5x5x5 voxels)
};

layout(std430, binding = 2) buffer MaterialIndices {
    uint materialIndices[]; // 4 materials per uint
};

layout(std430, binding = 3) buffer Materials {
    Material materials[];   // Array of material properties
};

// Convert 3D coordinates to local index for 5x5x5 structure
int getLocalIndex(ivec3 pos) {
    return pos.x + pos.z * BLOCK_SIZE + pos.y * BLOCK_SIZE * BLOCK_SIZE;
}

// Get chunk index from world position
int getChunkIndex(ivec3 pos) {
    return pos.x + pos.z * GRID_SIZE + pos.y * GRID_SIZE * GRID_SIZE;
}

// Get local position within a chunk/block
ivec3 getLocalPos(ivec3 pos) {
    return pos % BLOCK_SIZE;
}

// Get chunk position from chunk index
ivec3 getChunkPos(int index) {
    int x = index % GRID_SIZE;
    int z = (index / GRID_SIZE) % GRID_SIZE;
    int y = index / (GRID_SIZE * GRID_SIZE);
    return ivec3(x, y, z);
}

vec3 composePosition(ivec3 chunkPos, ivec3 blockPos, ivec3 voxelPos) {
    return vec3(chunkPos) + vec3(blockPos) / float(BLOCK_SIZE) + vec3(voxelPos) / float(CHUNK_SIZE);
}

void decomposePosition(vec3 pos, out ivec3 chunkPos, out ivec3 blockPos, out ivec3 voxelPos) {
    chunkPos = ivec3(floor(pos));
    blockPos = ivec3(floor(pos * BLOCK_SIZE)) % BLOCK_SIZE;
    voxelPos = ivec3(floor(pos * CHUNK_SIZE)) % BLOCK_SIZE;
}

int globalIndex(int parentIndex, int localIndex) {
    return parentIndex * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + localIndex;
}

// Bit manipulation for uvec4 masks
bool getBit(uvec4 mask, int index) {
    int uintIndex = index / 32;
    int bitIndex = index % 32;
    return (mask[uintIndex] & (1u << bitIndex)) != 0u;
}

void atomicSetChunkBit(int chunkIndex, int blockIndex) {
    int uintIndex = blockIndex / 32;
    int bitIndex = blockIndex % 32;
    atomicOr(chunkMasks[chunkIndex][uintIndex], 1u << bitIndex);
}

void atomicSetBlockBit(int blockIndex, int voxelIndex) {
    int uintIndex = voxelIndex / 32;
    int bitIndex = voxelIndex % 32;
    atomicOr(blockMasks[blockIndex][uintIndex], 1u << bitIndex);
}

bool isEmptyMask(uvec4 mask) {
    return mask.x == 0u && mask.y == 0u && mask.z == 0u && mask.w == 0u;
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













// DDA

// DDA State for raymarching
struct DDAState {
    ivec3 cell;
    ivec3 raySign;
    vec3 deltaDist;
    vec3 sideDist;
};

// Initialize DDA state
DDAState initDDA(vec3 pos, vec3 rayDir, vec3 invDir) {
    DDAState state;
    
    state.raySign = ivec3(sign(rayDir));
    state.cell = ivec3(floor(pos));
    state.deltaDist = invDir;
    state.sideDist = ((state.cell - pos) + 0.5 + state.raySign * 0.5) * state.deltaDist;

    return state;
}

vec3 stepMask(vec3 sideDist) {
    // Yoinked from https://www.shadertoy.com/view/l33XWf
    bvec3 move;
    bvec3 pon=lessThan(sideDist.xyz,sideDist.yzx);

    move.x=pon.x && !pon.z;
    move.y=pon.y && !pon.x;
    move.z=!(move.x||move.y);

    return vec3(move);
}

// Step DDA state and return normal
ivec3 stepDDA(inout DDAState state)
{
    ivec3 mask = ivec3(stepMask(state.sideDist));
    ivec3 normalNeg = mask * state.raySign;
    state.cell += normalNeg;
    state.sideDist += mask * state.raySign * state.deltaDist;
    return -normalNeg;
}

// Step DDA state without returning normal
void stepDDANoNormal(inout DDAState state) {
    ivec3 mask = ivec3(stepMask(state.sideDist));
    state.cell += mask * state.raySign;
    state.sideDist += vec3(mask) * state.deltaDist;
}

// Check if position is within bounds
bool isInBounds(ivec3 pos, ivec3 bounds) {
    return all(greaterThanEqual(pos, ivec3(EPSILON))) && 
           all(lessThan(pos, bounds));
}

// Ray-box intersection test
bool intersectBox(vec3 origin, vec3 invDir, vec3 boxMin, vec3 boxMax, 
                 out float tMin, out float tMax) {
    vec3 t0 = (boxMin - origin) * invDir;
    vec3 t1 = (boxMax - origin) * invDir;
    vec3 tminv = min(t0, t1);
    vec3 tmaxv = max(t0, t1);
    tMin = max(max(tminv.x, tminv.y), tminv.z);
    tMax = min(min(tmaxv.x, tmaxv.y), tmaxv.z);
    return tMin <= tMax && tMax > 0.0;
}

// Get intersection point with current cell
vec3 getDDAUVs(DDAState state, vec3 rayPos, vec3 rayDir) {
    vec3 mini = (vec3(state.cell) - rayPos + 0.5 - 0.5 * vec3(state.raySign)) * state.deltaDist;
    float d = max(mini.x, max(mini.y, mini.z));
    vec3 intersect = rayPos + rayDir * d;
    vec3 uvs = intersect - vec3(state.cell);
    
    if (state.cell == ivec3(floor(rayPos))) {
        uvs = rayPos - vec3(state.cell);
    }
    
    return uvs;
}











// Hash map implementation for block to material index

// Constants 
const uint EMPTY_KEY = 0xFFFFFFFFu; // all table entries need to be initialized to this in C#
const uint EMPTY_VALUE = 0xFFFFFFFFu;
const uint TABLE_SIZE = 536870912; // Must be power of 2
const uint TABLE_MASK = TABLE_SIZE - 1u;
const uint maxProbes = 128u; // Prevent infinite loops

struct KeyValue {
    uint blockIndex;
    uint matStartIdx; // will be divided by 4 to get array index and use mod to find the offset
};

layout(std430, binding = 4) buffer BlocksHashMap {
    KeyValue blockToMatPtr[];
};
// voxel materials buffer
layout(std430, binding = 5) buffer VoxelMaterials {
    uint voxelMaterials[]; // 4 materials per uint
};
layout(std430, binding = 6) buffer VoxelMaterialsCounter {
    uint voxelMaterialsCounter;
};


// MurmurHash3 implementation
uint hash(uint k)
{
    k ^= k >> 16;
    k *= 0x85ebca6bu;
    k ^= k >> 13;
    k *= 0xc2b2ae35u;
    k ^= k >> 16;
    return k & TABLE_MASK;
}

// Returns true if insert successful
bool insertBlockKey(uint blockIndex, out uint outMatStartIdx) {
    uint slot = hash(blockIndex);
    uint probeCount = 0u;
    
    while (probeCount < maxProbes) {
        // Try atomic compare and swap on the blockIndex
        uint prevKey = atomicCompSwap(blockToMatPtr[slot].blockIndex, EMPTY_KEY, blockIndex);
        
        if (prevKey == EMPTY_KEY) {
            // We claimed the slot, so we increment the counter
            uint newMatStartIdx = atomicAdd(voxelMaterialsCounter, 1);
            // Set matStartIdx
            atomicExchange(blockToMatPtr[slot].matStartIdx, newMatStartIdx);
            outMatStartIdx = newMatStartIdx;
            return true;
        }
        
        if (prevKey == blockIndex) {
            // Slot already exists, return its matStartIdx
            outMatStartIdx = blockToMatPtr[slot].matStartIdx;
            return true;
        }
        
        // Linear probe
        slot = (slot + 1u) & TABLE_MASK;
        probeCount++;
    }
    
    return false; // Table full or too many collisions
}

uint lookupBlockKey(uint blockIndex) {
    uint slot = hash(blockIndex);
    uint probeCount = 0u;
    
    while (probeCount < maxProbes) {
        uint k = blockToMatPtr[slot].blockIndex;
        if (k == blockIndex) {
            return blockToMatPtr[slot].matStartIdx;
        }
        if (k == EMPTY_KEY) {
            return EMPTY_VALUE;
        }
        slot = (slot + 1u) & TABLE_MASK;
        probeCount++;
    }
    
    return EMPTY_VALUE;
}

void deleteBlockKey(uint blockIndex) {
    uint slot = hash(blockIndex);
    uint probeCount = 0u;
    
    while (probeCount < maxProbes) {
        if (blockToMatPtr[slot].blockIndex == blockIndex) {
            // Mark as deleted by setting matStartIdx to empty
            atomicExchange(blockToMatPtr[slot].matStartIdx, EMPTY_VALUE);
            return;
        }
        if (blockToMatPtr[slot].blockIndex == EMPTY_KEY) {
            return;
        }
        slot = (slot + 1u) & TABLE_MASK;
        probeCount++;
    }
}


// Material functions
void atomicSetMaterialIndex(int blockIndex, uint materialIndex)
{
    int arrayPosition = blockIndex / 4;
    int packedPosition = blockIndex % 4;
    
    // unset the bits
    atomicAnd(materialIndices[arrayPosition], ~(0xFFu << (packedPosition * 8)));
    // set the bits
    atomicOr(materialIndices[arrayPosition], materialIndex << (packedPosition * 8));
}

bool isBlockHomogenous(int blockIndex)
{
    // check the 126th bit of the mask
    return (atomicOr(blockMasks[blockIndex].w, 0) & 0x20000000u) != 0u;
}

bool isBlockInitialized(int blockIndex)
{
    // check the 127th bit of the mask
    return (atomicOr(blockMasks[blockIndex].w, 0) & 0x40000000u) != 0u;
}

void setBlockHomogenous(int blockIndex)   
{
    atomicOr(blockMasks[blockIndex].w, 0x20000000u);
}

void setBlockInitialized(int blockIndex)
{
    atomicOr(blockMasks[blockIndex].w, 0x40000000u);
}

void clearBlockHomogenous(int blockIndex)
{
    atomicAnd(blockMasks[blockIndex].w, ~0x20000000u);
}

void clearBlockInitialized(int blockIndex)
{
    atomicAnd(blockMasks[blockIndex].w, ~0x40000000u);
}

uint getBlockMaterialIndex(int blockIndex)
{
    int arrayPosition = blockIndex / 4;
    int packedPosition = blockIndex % 4;
    return uint((atomicOr(materialIndices[arrayPosition], 0) >> (packedPosition * 8)) & 0xFFu);
}

void _setVoxelMaterial(int blockIndex, int voxelLocalIndex, uint materialIndex) {
    // First try to get existing entry
    uint matStartIdx;
    insertBlockKey(blockIndex, matStartIdx);

    // Now set the material index
    uint arrayPosition = uint(matStartIdx * (125.0 / 4.0)) + voxelLocalIndex / 4;
    uint packedPosition = voxelLocalIndex % 4;

    atomicAnd(voxelMaterials[arrayPosition], ~ (0x000000FFu << (packedPosition * 8)));
    atomicOr(voxelMaterials[arrayPosition], materialIndex << (packedPosition * 8));
}

void setVoxelMaterial(int blockIndex, int voxelLocalIndex, uint materialIndex)
{
    bool isInited = isBlockInitialized(blockIndex);

    // if the material has not been set yet, set the homogenous bit
    if (!isInited)
    {
        setBlockHomogenous(blockIndex);
        setBlockInitialized(blockIndex);
        atomicSetMaterialIndex(blockIndex, materialIndex);
    }

    bool isHomogenous = isBlockHomogenous(blockIndex);
    uint blockMaterial = getBlockMaterialIndex(blockIndex);

    // if the material has been set already and is different, unset the homogenous bit
    if (isInited && isHomogenous && blockMaterial != materialIndex)
    {
        clearBlockHomogenous(blockIndex);
    }

    isHomogenous = isBlockHomogenous(blockIndex);
    
    // if the block is not homogenous then we need to set the voxel data too
    if(!isHomogenous)
    {
        // if the material is not homogenous, set the voxel material
        _setVoxelMaterial(blockIndex, voxelLocalIndex, materialIndex);
    }
} 

uint getVoxelMaterial(int blockIndex, int voxelLocalIndex)
{
    uint matStartIdx = lookupBlockKey(blockIndex);
    if (matStartIdx == EMPTY_VALUE)
    {
        return EMPTY_VALUE;
    }

    uint arrayPosition = uint(matStartIdx * (125.0 / 4.0)) + voxelLocalIndex / 4;
    uint packedPosition = voxelLocalIndex % 4;
    return uint((voxelMaterials[arrayPosition] >> (packedPosition * 8)) & 0x000000FFu);
}