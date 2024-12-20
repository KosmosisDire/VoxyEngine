#version 460

const float PI = 3.14159265359;

// Core Constants
const float EPSILON = 0.001;
const int MAX_STEPS = 256;
const float MAX_DIST = 1000.0;

// Structure Constants
const int BLOCK_SIZE = 8;      
const int CHUNK_SIZE = BLOCK_SIZE * BLOCK_SIZE;
const int GRID_SIZE = 25;          
const float VOXEL_SIZE = 1.0 / CHUNK_SIZE;
const uint INVALID_INDEX = 4294967295u;

const uint BITS_PER_UINT = 32u;
const uint BITS_PER_ARRAY = 128u;
const uint UINTS_PER_MASK = 4u;

const int NUM_DIRECTIONS = 26;
const vec3 NORM_DIRECTIONS[26] = vec3[26](
    // Must match C# MAIN_DIRECTIONS exactly
    vec3(1, 0, 0), vec3(-1, 0, 0),
    vec3(0, 1, 0), vec3(0, -1, 0),
    vec3(0, 0, 1), vec3(0, 0, -1),
    
    normalize(vec3(1, 1, 0)), normalize(vec3(-1, 1, 0)),
    normalize(vec3(1, -1, 0)), normalize(vec3(-1, -1, 0)),
    normalize(vec3(1, 0, 1)), normalize(vec3(-1, 0, 1)),
    normalize(vec3(1, 0, -1)), normalize(vec3(-1, 0, -1)),
    normalize(vec3(0, 1, 1)), normalize(vec3(0, -1, 1)),
    normalize(vec3(0, 1, -1)), normalize(vec3(0, -1, -1)),
    
    normalize(vec3(1, 1, 1)), normalize(vec3(-1, 1, 1)),
    normalize(vec3(1, -1, 1)), normalize(vec3(-1, -1, 1)),
    normalize(vec3(1, 1, -1)), normalize(vec3(-1, 1, -1)),
    normalize(vec3(1, -1, -1)), normalize(vec3(-1, -1, -1))
);

// Material System
struct Material
{
    // gradient colors
    vec4 colors[4];

    float noiseSize;
    float noiseStrength;
    uint textureIds[2];
    float textureBlend;

    bool blendWithNoise;
    bool isPowder;
    bool isLiquid;
    bool isCollidable;

    float shininess;
    float specularStrength;
    float reflectivity;
    float transparency;
    float refractiveIndex;
    float emission;

    float p1;
};

// Buffer Bindings
layout(std430, binding = 0) buffer ChunkMasks {
    uvec4 chunkMasks[];    // 512 bits per chunk for block occupancy (4 uvec4s per block)
};

layout(std430, binding = 1) buffer BlockMasks {
    uvec4 blockMasks[];    // 512 bits per block for voxel occupancy (4 uvec4s per voxel)
};

layout(std430, binding = 2) buffer MaterialIndices {
    uint materialIndices[]; // 4 materials per uint
};

layout(std430, binding = 3) buffer Materials {
    Material materials[];   // Array of material properties
};

layout(std430, binding = 4) buffer UncompressedMaterials {
    uint uncompressedMaterials[];   // 4 materials per uint. One material index per voxel for use during voxel modifications before compression
};

// Convert 3D coordinates to local index
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
    return ivec3(index % GRID_SIZE, index / (GRID_SIZE * GRID_SIZE), (index / GRID_SIZE) % GRID_SIZE);
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

void atomicSetChunkBit(int chunkIndex, int blockIndex) {
    uint baseIndex = chunkIndex << 2u;  // chunkIndex * 4 (faster than multiplication)
    uint arrayIndex = baseIndex + (blockIndex >> 7u); // blockIndex / 128
    uint uintIndex = (blockIndex >> 5u) & 3u; // (blockIndex / 32) % 4
    uint bitIndex = blockIndex & 31u; // blockIndex % 32
    
    // Pre-shift the bit to its final position
    uint bitMask = 1u << bitIndex;
    
    // Single atomic operation
    atomicOr(chunkMasks[arrayIndex][uintIndex], bitMask);
}

void atomicSetBlockBit(int blockIndex, int voxelIndex) {
    uint baseIndex = blockIndex << 2u;  // blockIndex * 4
    uint arrayIndex = baseIndex + (voxelIndex >> 7u); // voxelIndex / 128
    uint uintIndex = (voxelIndex >> 5u) & 3u; // (voxelIndex / 32) % 4
    uint bitIndex = voxelIndex & 31u; // voxelIndex % 32
    
    // Pre-shift the bit to its final position
    uint bitMask = 1u << bitIndex;
    
    atomicOr(blockMasks[arrayIndex][uintIndex], bitMask);
}

struct Bitmask {
    uvec4 mask1;
    uvec4 mask2;
    uvec4 mask3;
    uvec4 mask4;
};

// Bit manipulation for masks
bool getBit(Bitmask mask, int index) {
    uvec4 selectedArray = 
        (index < 128) ? mask.mask1 :
        (index < 256) ? mask.mask2 :
        (index < 384) ? mask.mask3 :
                       mask.mask4;
                       
    return (selectedArray[(index >> 5) & 3] & (1u << (index & 31))) != 0u;
}

Bitmask getChunkBitmask(int chunkIndex) {
    Bitmask mask;
    int arrayIndex = chunkIndex * 4;
    mask.mask1 = chunkMasks[arrayIndex];
    mask.mask2 = chunkMasks[arrayIndex + 1];
    mask.mask3 = chunkMasks[arrayIndex + 2];
    mask.mask4 = chunkMasks[arrayIndex + 3];
    return mask;
}

Bitmask getBlockBitmask(int blockIndex) {
    Bitmask mask;
    int arrayIndex = blockIndex * 4;
    mask.mask1 = blockMasks[arrayIndex];
    mask.mask2 = blockMasks[arrayIndex + 1];
    mask.mask3 = blockMasks[arrayIndex + 2];
    mask.mask4 = blockMasks[arrayIndex + 3];
    return mask;
}

bool isEmptyMask(Bitmask mask) {
    return mask.mask1 == uvec4(0u) && mask.mask2 == uvec4(0u) && mask.mask3 == uvec4(0u) && mask.mask4 == uvec4(0u);
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

void setMaterial(int voxelIndex, uint materialIndex) {
    uint uintIndex = voxelIndex / 4;  // Each uint stores 4 8-bit values
    uint bitPosition = (voxelIndex % 4) * 8;  // Each material takes 8 bits
    
    // Clear the byte at the target position (8 bits)
    uint clearMask = ~(0xFFu << bitPosition);
    // Set the new material value
    uint setMask = (materialIndex & 0xFFu) << bitPosition;
    
    atomicAnd(uncompressedMaterials[uintIndex], clearMask);
    atomicOr(uncompressedMaterials[uintIndex], setMask);
}

uint getMaterial(int voxelIndex) {
    uint uintIndex = voxelIndex / 4;
    uint bitPosition = (voxelIndex % 4) * 8;
    return (uncompressedMaterials[uintIndex] >> bitPosition) & 0xFFu;
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

float sampleOccupancy(ivec3 pos) {
    if (any(lessThan(pos, ivec3(0))) || any(greaterThanEqual(pos, ivec3(CHUNK_SIZE * GRID_SIZE)))) {
        return 0.0; // Outside grid is empty
    }
    
    ivec3 chunkPos = pos / CHUNK_SIZE;
    int chunkIndex = getChunkIndex(chunkPos);
    Bitmask chunkMask = getChunkBitmask(chunkIndex);
    
    if (isEmptyMask(chunkMask)) {
        return 0.0;
    }
    
    ivec3 blockInChunk = (pos / BLOCK_SIZE) % BLOCK_SIZE;
    int blockIndex = getLocalIndex(blockInChunk);
    
    if (!getBit(chunkMask, blockIndex)) {
        return 0.0;
    }
    
    int globalBlockIndex = globalIndex(chunkIndex, blockIndex);
    Bitmask blockMask = getBlockBitmask(globalBlockIndex);
    ivec3 voxelInBlock = pos % BLOCK_SIZE;
    int voxelIndex = getLocalIndex(voxelInBlock);
    
    return getBit(blockMask, voxelIndex) ? 1.0 : 0.0;
}

vec3 getPerVoxelNormal(ivec3 cell, int voxelIndex, int blockIndex, Bitmask blockMask) {    
    // For interior voxels, use fast path
    ivec3 localPos = ivec3(
        voxelIndex % BLOCK_SIZE,
        voxelIndex / (BLOCK_SIZE * BLOCK_SIZE),
        (voxelIndex / BLOCK_SIZE) % BLOCK_SIZE
    );
    
    bool nearBoundary = any(lessThan(localPos, ivec3(1))) || 
                       any(greaterThanEqual(localPos, ivec3(BLOCK_SIZE - 1)));
                       
    if (!nearBoundary) {
        vec3 normal = vec3(0.0);
        bool hasEmptyNeighbor = false;
        
        if (!getBit(blockMask, getLocalIndex(localPos + ivec3(1,0,0)))) {
            normal.x += 1.0;
            hasEmptyNeighbor = true;
        }
        if (!getBit(blockMask, getLocalIndex(localPos + ivec3(-1,0,0)))) {
            normal.x -= 1.0;
            hasEmptyNeighbor = true;
        }
        if (!getBit(blockMask, getLocalIndex(localPos + ivec3(0,1,0)))) {
            normal.y += 1.0;
            hasEmptyNeighbor = true;
        }
        if (!getBit(blockMask, getLocalIndex(localPos + ivec3(0,-1,0)))) {
            normal.y -= 1.0;
            hasEmptyNeighbor = true;
        }
        if (!getBit(blockMask, getLocalIndex(localPos + ivec3(0,0,1)))) {
            normal.z += 1.0;
            hasEmptyNeighbor = true;
        }
        if (!getBit(blockMask, getLocalIndex(localPos + ivec3(0,0,-1)))) {
            normal.z -= 1.0;
            hasEmptyNeighbor = true;
        }
        
        if (hasEmptyNeighbor) {
            return normalize(normal);
        }
        
        return vec3(0.0); // Return zero normal if no empty neighbors
    }
    
    // For boundary voxels, use a separable gradient approximation
    const float kernel[3] = float[3](-0.5, 0.0, 0.5);
    const ivec3 offsets[3] = ivec3[3](ivec3(-1), ivec3(0), ivec3(1));
    
    vec3 gradient = vec3(0.0);
    
    // X gradient
    float xSum = 0.0;
    for (int i = 0; i < 3; i++) {
        ivec3 samplePos = cell + ivec3(offsets[i].x, 0, 0);
        xSum += kernel[i] * sampleOccupancy(samplePos);
    }
    gradient.x = xSum;
    
    // Y gradient
    float ySum = 0.0;
    for (int i = 0; i < 3; i++) {
        ivec3 samplePos = cell + ivec3(0, offsets[i].y, 0);
        ySum += kernel[i] * sampleOccupancy(samplePos);
    }
    gradient.y = ySum;
    
    // Z gradient
    float zSum = 0.0;
    for (int i = 0; i < 3; i++) {
        ivec3 samplePos = cell + ivec3(0, 0, offsets[i].z);
        zSum += kernel[i] * sampleOccupancy(samplePos);
    }
    gradient.z = zSum;
    
    // Return zero normal if gradient is too small, otherwise return normalized negative gradient
    return length(gradient) > EPSILON ? normalize(-gradient) : vec3(0.0);
}