uniform int chunkSize;
uniform int maxChunkCount;

struct ChunkData {
    vec3 position;       // Chunk world position
    float padding1;      // Padding for alignment
    int nodeCount;       // Number of nodes in the octree
    int leafCount;       // Number of leaf nodes
    int chunkIndex;    // Offset into the data buffers
    int padding2;        // Padding for alignment
};

layout(std430, binding = 0) buffer PositionsBuffer { int positions[]; }; // packed NodeData bytes 4x7 bits, 1x7 bits, 1x4 bits
layout(std430, binding = 1) buffer MaterialsBuffer { int materials[]; }; // packed VoxelMaterial bytes, 3x8 bits, 1x1 bit, 1x3 bits
layout(std430, binding = 2) buffer LightLevelsBuffer { vec4 lightLevels[]; };
layout(std430, binding = 4) readonly buffer ChunkDataBuffer {
    ChunkData chunks[];
};

struct NodeData {
    ivec3 position;  // 7 bits each (0-127)
    int size;        // 7 bits (0-127)
    int data;        // 4 bits (0-15) // 0 = not a leaf, 1 = leaf
};

NodeData unpackNodeData(int index)
{
    int packedValue = positions[index];
    NodeData node;
    
    // Extract position components (7 bits each)
    node.position.x = (packedValue >> 25) & 0x7F;
    node.position.y = (packedValue >> 18) & 0x7F;
    node.position.z = (packedValue >> 11) & 0x7F;
    
    // Extract size (7 bits)
    node.size = (packedValue >> 4) & 0x7F;
    
    // Extract data (4 bits)
    node.data = packedValue & 0xF;
    
    return node;
}

int packNodeData(NodeData node) {
    return (node.position.x << 25) | (node.position.y << 18) | (node.position.z << 11) | (node.size << 4) | node.data;
}

struct VoxelMaterial
{
    vec3 color;
    bool isSolid;
    int data;
};

VoxelMaterial unpackMaterial(int index)
{
    int packedValue = materials[index];
    VoxelMaterial material;
    
    // Extract color components (8 bits each)
    material.color.x = float((packedValue >> 24) & 0xFF) / 255.0;
    material.color.y = float((packedValue >> 16) & 0xFF) / 255.0;
    material.color.z = float((packedValue >> 8) & 0xFF) / 255.0;
    
    // Extract isSolid (1 bit)
    material.isSolid = ((packedValue >> 7) & 0x1) == 1;
    
    // Extract data (3 bits)
    material.data = packedValue & 0x7;
    
    return material;
}

int packMaterial(VoxelMaterial material) {
    return (int(material.color.x * 255) << 24) | (int(material.color.y * 255) << 16) | (int(material.color.z * 255) << 8) | (int(material.isSolid) << 7) | material.data;
}

float min3(vec3 v) {
    return min(min(v.x, v.y), v.z);
}

float max3(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

float distSquared(vec3 A, vec3 B) {
    vec3 C = A - B;
    return dot(C, C);
}

// Random number generation
uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

float random(uint seed) {
    return float(hash(seed)) / float(0xffffffffU);
}

vec3 getChunkPosition(vec3 worldPos) {
    float chunkWorldSize = chunkSize * voxelSize;
    return vec3(
        floor(worldPos.x / chunkWorldSize) * chunkWorldSize,
        floor(worldPos.y / chunkWorldSize) * chunkWorldSize,
        floor(worldPos.z / chunkWorldSize) * chunkWorldSize
    );
}

int findChunkIndex(vec3 worldPos) {
    vec3 chunkWorldSize = vec3(chunkSize * voxelSize);
    
    for (int i = 0; i < maxChunkCount; i++) {
        vec3 chunkMin = chunks[i].position;
        vec3 chunkMax = chunkMin + chunkWorldSize;
        
        if (worldPos.x >= chunkMin.x && worldPos.x < chunkMax.x &&
            worldPos.y >= chunkMin.y && worldPos.y < chunkMax.y &&
            worldPos.z >= chunkMin.z && worldPos.z < chunkMax.z) {
            return i;
        }
    }
    return -1;
}

int getDataIndex(ivec3 localPos, int chunkIndex) {
    return chunkIndex * chunkSize * chunkSize * chunkSize + localPos.x + localPos.y * chunkSize + localPos.z * chunkSize * chunkSize;
}


int getIndexFromLocalPosition(ivec3 pos)
{
    return pos.x + (pos.y << int(log2(float(chunkSize)))) + 
           (pos.z << int(log2(float(chunkSize * chunkSize))));
}

vec3 getLocalPositionFromIndex(int index) {
    int x = index % chunkSize;
    int y = (index / chunkSize) % chunkSize;
    int z = index / (chunkSize * chunkSize);
    return vec3(x, y, z) * voxelSize;
}

// Returns the index into materials/lightLevels arrays for a given world position
// Returns -1 if position is invalid or outside any chunk
int getVoxelAt(vec3 worldPos) {
    // Find which chunk contains this position
    int chunkIndex = findChunkIndex(worldPos);
    if (chunkIndex == -1) return -1;
    
    ChunkData chunk = chunks[chunkIndex];
    
    // Convert world position to local chunk coordinates
    vec3 localPos = (worldPos - chunk.position) / voxelSize;
    ivec3 voxelPos = ivec3(localPos);
    
    // Calculate the leaf node index
    int leafIndex = getIndexFromLocalPosition(voxelPos);

    // Convert to global buffer index
    return chunk.chunkIndex * chunk.leafCount + leafIndex;
}

int getDataIndexSafe(ivec3 localPos, int chunkIndex)
{
    if (localPos.x < 0 || localPos.x >= chunkSize ||
        localPos.y < 0 || localPos.y >= chunkSize ||
        localPos.z < 0 || localPos.z >= chunkSize) {
        return getVoxelAt(vec3(localPos * voxelSize + chunks[chunkIndex].position));
    }
    return chunkIndex * chunkSize * chunkSize * chunkSize + localPos.x + localPos.y * chunkSize + localPos.z * chunkSize * chunkSize;
}


