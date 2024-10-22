using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Util;
using System.Numerics;
namespace VoxelEngine;

public class GPULightingCalculator : IDisposable
{
    private Context context;
    private Accelerator accelerator;
    private Action<Index1D, ArrayView<Float4>, ArrayView<int>, ArrayView<float>, Vector3, int, float, long> directKernel;
    private Action<Index1D, ArrayView<int>, ArrayView<float>, int, float> blurKernel;

    MemoryBuffer1D<Float4, Stride1D.Dense> devicePosAndSize;
    MemoryBuffer1D<int, Stride1D.Dense> deviceMaterials;
    MemoryBuffer1D<float, Stride1D.Dense> deviceLightLevels;

    private OctreeGenerator octree;

    public GPULightingCalculator(OctreeGenerator octree)
    {
        context = Context.Create(builder => builder.Default().EnableAlgorithms().Optimize(OptimizationLevel.Release));
        accelerator = context.GetPreferredDevice(preferCPU: false)
                             .CreateAccelerator(context);

        directKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Float4>, ArrayView<int>, ArrayView<float>, Vector3, int, float, long>(DirectLightingKernel);
        blurKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<float>, int, float>(BlurKernel);

        devicePosAndSize = accelerator.Allocate1D<Float4>(octree.positions.Length);
        deviceMaterials = accelerator.Allocate1D<int>(octree.materials.Length);
        deviceLightLevels = accelerator.Allocate1D<float>(octree.lightLevels.Length);

        this.octree = octree;
    }

    public void UploadOctreeData()
    {
        devicePosAndSize.CopyFromCPU(octree.positions);
        deviceMaterials.CopyFromCPU(octree.materials);
        deviceLightLevels.CopyFromCPU(octree.lightLevels);
    }

    public void CalculateLighting(Vector3 lightDirection)
    {
        lightDirection = Vector3.Normalize(lightDirection);

        var startTime = DateTime.Now;

        // Launch the kernel
        directKernel(octree.NodeCount, devicePosAndSize.View, deviceMaterials.View, deviceLightLevels.View, lightDirection, octree.ChunkSize, octree.voxelSize, DateTime.UtcNow.Ticks);
        blurKernel(octree.LeafCount, deviceMaterials.View, deviceLightLevels.View, octree.ChunkSize, octree.voxelSize);
        blurKernel(octree.LeafCount, deviceMaterials.View, deviceLightLevels.View, octree.ChunkSize, octree.voxelSize);
        blurKernel(octree.LeafCount, deviceMaterials.View, deviceLightLevels.View, octree.ChunkSize, octree.voxelSize);

        // Copy the result back to the host
        deviceLightLevels.CopyToCPU(octree.lightLevels);
        
        var endTime = DateTime.Now;
        Console.WriteLine($"Lighting calculation took: {(endTime - startTime).TotalMilliseconds}ms");
    }

    public static void DirectLightingKernel(
    Index1D index,
    ArrayView<Float4> positions,
    ArrayView<int> materials,
    ArrayView<float> lightLevels,
    Vector3 lightDirection,
    int chunkSize,
    float voxelSize,
    long time
    )
    {
        // Get the current position and validate size
        Float4 positionSize = positions[index];
        if (XMath.Abs(positionSize.W) != voxelSize)
        {
            return;
        }

        Vector3 position = new(positionSize.X, positionSize.Y, positionSize.Z);
        int dataIndex = GetIndexFromPosition(position, chunkSize, voxelSize);

        // Compute neighbor indices once
        int indexMinusOne = dataIndex - 1;
        int indexPlusOne = dataIndex + 1;
        int indexMinusChunk = dataIndex - chunkSize;
        int indexPlusChunk = dataIndex + chunkSize;
        int indexMinusChunkSquared = dataIndex - chunkSize * chunkSize;
        int indexPlusChunkSquared = dataIndex + chunkSize * chunkSize;

        // Check neighbors more efficiently
        bool isNextToSolid = false;
        if (indexMinusOne >= 0 && materials[indexMinusOne] != 0) isNextToSolid = true;
        else if (indexPlusOne < lightLevels.Length && materials[indexPlusOne] != 0) isNextToSolid = true;
        else if (indexMinusChunk >= 0 && materials[indexMinusChunk] != 0) isNextToSolid = true;
        else if (indexPlusChunk < lightLevels.Length && materials[indexPlusChunk] != 0) isNextToSolid = true;
        else if (indexMinusChunkSquared >= 0 && materials[indexMinusChunkSquared] != 0) isNextToSolid = true;
        else if (indexPlusChunkSquared < lightLevels.Length && materials[indexPlusChunkSquared] != 0) isNextToSolid = true;

        if (!isNextToSolid)
        {
            return;
        }

        // Cast position calculation
        uint seed = (uint)(index * 31 + (time % 1000));
        seed = WangHash(seed);
        Vector3 castPosition = position + new Vector3(0.1f) + new Vector3(voxelSize * 0.8f) * (seed / uint.MaxValue);

        float lightLevel = 0;

        // Primary ray cast
        int hit = RayOctreeIntersection(castPosition, lightDirection, 1000, chunkSize, voxelSize, positions, materials);

        if (hit == -1)
        {
            lightLevel = 1.0f;
        }
        else
        {
            const int samples = 0;
            for (int i = 0; i < samples; i++)
            {
                Vector3 scatterDirection = GetStratifiedDirection((uint)(seed + i * 31));
                Vector3 currentPos = castPosition;
                hit = RayOctreeIntersection(currentPos, scatterDirection, 100, chunkSize, voxelSize, positions, materials);

                if (hit != -1)
                {
                    var hitIndexPosition = positions[hit];
                    int hitIndex = GetIndexFromPosition(new Vector3(hitIndexPosition.X, hitIndexPosition.Y, hitIndexPosition.Z), chunkSize, voxelSize);
                    lightLevel += lightLevels[hitIndex];
                }
            }
        }

        const float BLEND_FACTOR = 0.95f;
        lightLevel = XMath.Clamp(lightLevel, 0.0f, 1.0f);
        lightLevels[dataIndex] = lightLevels[dataIndex] * (1.0f - BLEND_FACTOR) + lightLevel * BLEND_FACTOR;
    }

    // Wang hash for better random number generation
    private static uint WangHash(uint seed)
    {
        seed = (seed ^ 61) ^ (seed >> 16);
        seed *= 9;
        seed = seed ^ (seed >> 4);
        seed *= 0x27d4eb2d;
        seed = seed ^ (seed >> 15);
        return seed;
    }

    // Stratified sampling for better direction generation
    private static Vector3 GetStratifiedDirection(uint seed)
    {
        float u = (float)((seed & 0xFFFF) / 65536.0f);
        float v = (float)(((seed >> 16) & 0xFFFF) / 65536.0f);

        float theta = 2.0f * XMath.PI * u;
        float phi = XMath.Acos(2.0f * v - 1.0f);

        float sinPhi = XMath.Sin(phi);
        return new Vector3(
            XMath.Cos(theta) * sinPhi,
            XMath.Sin(theta) * sinPhi,
            XMath.Cos(phi)
        );
    }

    public static void BlurKernel(
        Index1D index,
        ArrayView<int> materials,
        ArrayView<float> lightLevels,
        int chunkSize,
        float voxelSize
        )
    {
        int dataIndex = index;

        if (materials[dataIndex] != 0)
        {
            return;
        }


        int x = dataIndex % chunkSize;
        int y = (dataIndex / chunkSize) % chunkSize;
        int z = dataIndex / (chunkSize * chunkSize);

        float sum = 0.0f;
        int count = 0;

        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                for (int k = -1; k <= 1; k++)
                {
                    int neighborX = x + i;
                    int neighborY = y + j;
                    int neighborZ = z + k;

                    if (neighborX >= 0 && neighborX < chunkSize &&
                        neighborY >= 0 && neighborY < chunkSize &&
                        neighborZ >= 0 && neighborZ < chunkSize)
                    {
                        int neighborIndex = neighborX + neighborY * chunkSize + neighborZ * chunkSize * chunkSize;
                        // skip if the neighbor is solid
                        if (materials[neighborIndex] != 0)
                        {
                            continue;
                        }
                        sum += lightLevels[neighborIndex];
                        count++;
                    }
                }
            }
        }

        lightLevels[dataIndex] = sum / count;
    }

    // A simple random number generator suitable for GPU use
    private static int XorShift(ref int state)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }

    // Generate a random float between 0 and 1
    private static float RandomFloat(ref int state)
    {
        return (float)XorShift(ref state) / uint.MaxValue;
    }

    // Generate a random direction vector from a seed
    public static Vector3 GetRandomDirection(int seed)
    {
        int state = seed;

        // Generate random spherical coordinates
        float theta = RandomFloat(ref state) * 2 * XMath.PI;
        float phi = XMath.Acos(2 * RandomFloat(ref state) - 1);

        // Convert spherical coordinates to Cartesian coordinates
        float x = XMath.Sin(phi) * XMath.Cos(theta);
        float y = XMath.Sin(phi) * XMath.Sin(theta);
        float z = XMath.Cos(phi);

        return new Vector3(x, y, z);
    }

    private static bool IntersectBox(Vector3 origin, Vector3 invDir, Vector3 boxMin, Vector3 boxMax, out float tMin)
    {
        Vector3 t0 = (boxMin - origin) * invDir;
        Vector3 t1 = (boxMax - origin) * invDir;

        Vector3 tMinVec = Vector3.Min(t0, t1);
        Vector3 tMaxVec = Vector3.Max(t0, t1);

        tMin = XMath.Max(XMath.Max(tMinVec.X, tMinVec.Y), tMinVec.Z);
        float tMax = XMath.Min(XMath.Min(tMaxVec.X, tMaxVec.Y), tMaxVec.Z);

        return tMin <= tMax && tMax > 0.0f;
    }

    private static int GetIndexFromPosition(Vector3 position, int chunkSize, float voxelSize)
    {
        position = position / voxelSize;
        return (int)(position.X + position.Y * chunkSize + position.Z * chunkSize * chunkSize);
    }

    private static int RayOctreeIntersection(Vector3 origin, Vector3 dir, float maxDist, int chunkSize, float voxelSize, ArrayView<Float4> positions, ArrayView<int> materials)
    {
        Vector3 invDir = new Vector3(1.0f / dir.X, 1.0f / dir.Y, 1.0f / dir.Z);

        int[] stack = new int[64];
        int stackPtr = 0;
        stack[stackPtr++] = 0; // Start with root node

        while (stackPtr > 0)
        {
            int current = stack[--stackPtr];

            Float4 positionSize = positions[current];
            Vector3 position = new Vector3(positionSize.X, positionSize.Y, positionSize.Z);
            float size = positionSize.W;
            bool leaf = size < 0f;
            size = XMath.Abs(size);
            Vector3 nodeMin = position;
            Vector3 nodeMax = position + new Vector3(size);

            float tMin;
            if (!IntersectBox(origin, invDir, nodeMin, nodeMax, out tMin) || tMin > maxDist)
            {
                continue;
            }

            int dataIndex = GetIndexFromPosition(position, chunkSize, voxelSize);
            int material = materials[dataIndex];

            if (leaf)  // Hit a leaf node
            {
                if (material == 0) continue;
                return current;
            }
            else
            {
                int firstChildIndex = current * 8 + 1;

                // Precompute child order in a more efficient manner without sign() calls
                int childOrder = (dir.X > 0.0f ? 1 : 0) | (dir.Y > 0.0f ? 2 : 0) | (dir.Z > 0.0f ? 4 : 0);

                for (int i = 0; i < 8; ++i)
                {
                    stack[stackPtr++] = firstChildIndex + (childOrder ^ i);  // Efficient traversal
                }
            }
        }

        return -1;
    }

    public void Dispose()
    {
        accelerator?.Dispose();
        context?.Dispose();
        devicePosAndSize?.Dispose();
        deviceMaterials?.Dispose();
        deviceLightLevels?.Dispose();
    }
}