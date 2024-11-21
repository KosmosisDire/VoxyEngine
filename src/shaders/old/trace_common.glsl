// struct GlobalOctreeNode {
//     vec3 position;
//     float size;
//     int chunkIndex;
//     int padding1;
//     int padding2;
//     int padding3;
// };

// layout(std430, binding = 5) readonly buffer GlobalOctreeBuffer {
//     GlobalOctreeNode globalNodes[];
// };

// struct RayHit { 
//     int chunkIndex;
//     int nodeIndex;
//     vec3 point;
//     vec3 normal;
//     int steps;
// }; 

// bool intersectBox(vec3 origin, vec3 invDir, vec3 boxMin, vec3 boxMax, out float tMin, out float tMax)
// {
//     vec3 t0 = (boxMin - origin) * invDir;
//     vec3 t1 = (boxMax - origin) * invDir;
//     vec3 tMinVec = min(t0, t1);
//     vec3 tMaxVec = max(t0, t1);
//     tMin = max(max(tMinVec.x, tMinVec.y), tMinVec.z);
//     tMax = min(min(tMaxVec.x, tMaxVec.y), tMaxVec.z);
//     return tMin <= tMax && tMax > 0.0;
// }

// RayHit rayOctreeIntersection(vec3 origin, vec3 dir, float maxDist)
// {
//     vec3 invDir = 1.0 / dir;
    
//     RayHit hit;
//     hit.chunkIndex = -1;
//     hit.nodeIndex = -1;
//     hit.normal = vec3(0);
//     hit.steps = 0;
    
//     // Start with root node
//     int outerStack[64];
//     int outerPtr = 0;
//     outerStack[outerPtr++] = 0;

//     int childOrder = (dir.x > 0.0 ? 1 : 0) | (dir.y > 0.0 ? 2 : 0) | (dir.z > 0.0 ? 4 : 0);
    
//     while (outerPtr > 0 && hit.steps < 1024)
//     {
//         hit.steps++;
//         int nodeIndex = outerStack[--outerPtr];
//         GlobalOctreeNode node = globalNodes[nodeIndex];
//         bool isLeaf = node.size < 0.0;
//         float size = abs(node.size);
        
//         vec3 nodeMin = node.position;
//         vec3 nodeMax = nodeMin + vec3(size);
        
//         float tMin, tMax;
//         if (!intersectBox(origin, invDir, nodeMin, nodeMax, tMin, tMax) || tMin > maxDist)
//             continue;

//         // If this is a leaf node with a chunk
//         if (isLeaf && node.chunkIndex >= 0)
//         {
//             // Intersect with chunk's octree
//             ChunkData chunk = chunks[node.chunkIndex];
//             int maxNodes = chunk.nodeCount;
//             int baseOffset = chunk.chunkIndex * maxNodes;
//             vec3 chunkPosition = chunk.position;
            
//             // Start with chunk's root node
//             int innerStack[64];
//             int innerPtr = 0;

//             // start with the 8 children of tha parent node
//             int firstChildIndex = 1;
//             for (int i = 0; i < 8; ++i) {
//                 innerStack[innerPtr++] = firstChildIndex + (childOrder ^ i);
//             }
            
//             while (innerPtr > 0)
//             {
//                 hit.steps++;

//                 int current = innerStack[--innerPtr] + baseOffset;
//                 NodeData data = unpackNodeData(current);
//                 vec3 position = data.position + chunkPosition;
//                 float size = data.size;
//                 bool isLeaf = data.data == 1;

//                 vec3 nodeMin = position;
//                 vec3 nodeMax = position + vec3(size);

//                 float tMin, tMax;
//                 if (!intersectBox(origin, invDir, nodeMin, nodeMax, tMin, tMax) || tMin > maxDist) {
//                     continue;
//                 }

//                 if (isLeaf)
//                 {
//                     int dataIndex = getDataIndex(data.position, node.chunkIndex);
//                     VoxelMaterial material = unpackMaterial(dataIndex);

//                     if (!material.isSolid) continue;
                
//                     hit.chunkIndex = node.chunkIndex;
//                     hit.nodeIndex = current;
//                     hit.point = origin + dir * tMin;
                
//                     const float bias = 0.0001;
//                     if (abs(hit.point.x - nodeMin.x) < bias)
//                         hit.normal = vec3(-1.0, 0.0, 0.0);
//                     else if (abs(hit.point.x - nodeMax.x) < bias)
//                         hit.normal = vec3(1.0, 0.0, 0.0);
//                     else if (abs(hit.point.y - nodeMin.y) < bias)
//                         hit.normal = vec3(0.0, -1.0, 0.0);
//                     else if (abs(hit.point.y - nodeMax.y) < bias)
//                         hit.normal = vec3(0.0, 1.0, 0.0);
//                     else if (abs(hit.point.z - nodeMin.z) < bias)
//                         hit.normal = vec3(0.0, 0.0, -1.0);
//                     else if (abs(hit.point.z - nodeMax.z) < bias)
//                         hit.normal = vec3(0.0, 0.0, 1.0);
                    
//                     return hit;
//                 }

//                 int firstChildIndex = (current - baseOffset) * 8 + 1;
//                 for (int i = 0; i < 8; ++i) {
//                     innerStack[innerPtr++] = firstChildIndex + (childOrder ^ i);
//                 }
//             }
//         }
//         else if (!isLeaf)
//         {
//             int firstChildIndex = nodeIndex * 8 + 1;
//             for (int i = 0; i < 8; ++i) {
//                 outerStack[outerPtr++] = firstChildIndex + (childOrder ^ i);
//             }
//         }
//     }
    
//     return hit;
// }

struct GlobalOctreeNode {
    vec3 position;
    float size;
    int chunkIndex;
    int padding1;
    int padding2;
    int padding3;
};

layout(std430, binding = 5) readonly buffer GlobalOctreeBuffer {
    GlobalOctreeNode globalNodes[];
};


struct RayHit { 
    int chunkIndex;
    int nodeIndex;
    vec3 point;
    vec3 normal;
    int steps;
}; 

bool intersectBox(vec3 origin, vec3 invDir, vec3 boxMin, vec3 boxMax, out float tMin, out float tMax) {
    vec3 t0 = (boxMin - origin) * invDir;
    vec3 t1 = (boxMax - origin) * invDir;
    vec3 tMinVec = min(t0, t1);
    vec3 tMaxVec = max(t0, t1);
    tMin = max(max(tMinVec.x, tMinVec.y), tMinVec.z);
    tMax = min(min(tMaxVec.x, tMaxVec.y), tMaxVec.z);
    return tMin <= tMax && tMax > 0.0;
}

vec3 calculateNormal(vec3 point, vec3 dir, vec3 nodeMin, vec3 nodeMax) {
    vec3 normal = vec3(0.0);
    vec3 center = (nodeMin + nodeMax) * 0.5;
    vec3 size = nodeMax - nodeMin;
    
    // Calculate distance to each face plane
    vec3 d = abs(vec3(
        point.x - nodeMin.x < 0.001 ? 0.0 : point.x - nodeMax.x,
        point.y - nodeMin.y < 0.001 ? 0.0 : point.y - nodeMax.y,
        point.z - nodeMin.z < 0.001 ? 0.0 : point.z - nodeMax.z
    ));
    
    // Find closest face considering ray direction
    if(d.x < d.y && d.x < d.z) {
        normal = vec3(sign(-dir.x), 0.0, 0.0);
    } else if(d.y < d.z) {
        normal = vec3(0.0, sign(-dir.y), 0.0);
    } else {
        normal = vec3(0.0, 0.0, sign(-dir.z));
    }
    
    return normal;
}

RayHit rayOctreeIntersection(vec3 origin, vec3 dir, float maxDist) {
    RayHit hit;
    hit.chunkIndex = -1;
    hit.nodeIndex = -1;
    hit.normal = vec3(0);
    hit.steps = 0;
    
    vec3 invDir = 1.0 / max(abs(dir), vec3(1e-5));
    vec3 point = origin;
    float dist = 0.0;
    
    // Pre-calculate child traversal order based on ray direction
    int childOrder = (dir.x > 0.0 ? 1 : 0) | 
                    (dir.y > 0.0 ? 2 : 0) | 
                    (dir.z > 0.0 ? 4 : 0);
    
    // Start at root node
    int currentNodeIndex = 0;
    GlobalOctreeNode node = globalNodes[currentNodeIndex];
    
    while (dist < maxDist && hit.steps < 512) {
        hit.steps++;
        
        float nodeSize = abs(node.size);
        vec3 nodeMin = node.position;
        vec3 nodeMax = nodeMin + vec3(nodeSize);
        
        float tMin, tMax;
        if (!intersectBox(point, invDir, nodeMin, nodeMax, tMin, tMax)) {
            // Ray missed current node, move to next sibling or parent
            int parentIndex = (currentNodeIndex - 1) / 8;
            if (parentIndex < 0) break; // We're done if we've exhausted the tree
            
            // Skip to next sibling
            int childBase = parentIndex * 8 + 1;
            int localIndex = currentNodeIndex - childBase;
            localIndex = (localIndex + 1) & 7; // Wrap around to next sibling
            
            currentNodeIndex = childBase + (localIndex ^ childOrder);
            node = globalNodes[currentNodeIndex];
            continue;
        }
        
        // We're in a valid node
        if (node.chunkIndex >= 0) {
            // We're in a chunk node
            ChunkData chunk = chunks[node.chunkIndex];
            vec3 localPoint = point - chunk.position;
            ivec3 voxelPos = ivec3(floor(localPoint / voxelSize));
            
            // Check chunk bounds
            if (all(greaterThanEqual(voxelPos, ivec3(0))) && 
                all(lessThan(voxelPos, ivec3(chunkSize)))) {
                
                int dataIndex = getDataIndex(voxelPos, node.chunkIndex);
                VoxelMaterial material = unpackMaterial(dataIndex);
                
                if (material.isSolid) {
                    hit.chunkIndex = node.chunkIndex;
                    hit.nodeIndex = currentNodeIndex;
                    hit.point = point;
                    
                    vec3 voxelMin = vec3(voxelPos) * voxelSize + chunk.position;
                    vec3 voxelMax = voxelMin + vec3(voxelSize);
                    hit.normal = calculateNormal(point, dir, voxelMin, voxelMax);
                    
                    return hit;
                }
                
                // Step by voxel size within chunks
                point += dir * (voxelSize * 0.9);
                dist += voxelSize * 0.9;
            } else {
                // Step to chunk boundary
                point += dir * (tMin + voxelSize);
                dist += tMin + voxelSize;
                
                // Move back to root to find next chunk
                currentNodeIndex = 0;
                node = globalNodes[currentNodeIndex];
            }
        } else if (nodeSize < 0.0) {
            // Empty leaf node, skip it entirely
            point += dir * (tMax + voxelSize);
            dist += tMax + voxelSize;
            
            // Move back to root to find next node
            currentNodeIndex = 0;
            node = globalNodes[currentNodeIndex];
        } else {
            // Interior node, descend to first intersected child
            int firstChildIndex = currentNodeIndex * 8 + 1;
            bool foundChild = false;
            
            // Test children in traversal order
            for (int i = 0; i < 8; i++) {
                int childIndex = firstChildIndex + (i ^ childOrder);
                GlobalOctreeNode child = globalNodes[childIndex];
                float childSize = abs(child.size);
                
                vec3 childMin = child.position;
                vec3 childMax = childMin + vec3(childSize);
                
                float childTMin, childTMax;
                if (intersectBox(point, invDir, childMin, childMax, childTMin, childTMax) && childTMin < tMax) {
                    currentNodeIndex = childIndex;
                    node = child;
                    foundChild = true;
                    break;
                }
            }
            
            if (!foundChild) {
                // No valid children found, move to next sibling
                point += dir * (tMax + voxelSize);
                dist += tMax + voxelSize;
                currentNodeIndex = 0;
                node = globalNodes[currentNodeIndex];
            }
        }
    }
    
    return hit;
}