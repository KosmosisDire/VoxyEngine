using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using System.Numerics;
namespace VoxelEngine;

public class GPULightingCalculator : IDisposable
{
    private Context context;
    private Accelerator accelerator;
    private Action<Index1D, ArrayView<Vector4>, ArrayView<Vector4>, ArrayView<float>, Vector3, int, float, float, float, float> directKernel;
    private Action<Index1D, ArrayView<Vector4>, ArrayView<Vector4>, ArrayView<float>, Vector3, int, float, float, float, float> indirectKernel;
    private Action<Index1D, ArrayView<Vector4>, ArrayView<Vector4>, ArrayView<float>, Vector3, int, float, float, float, float> blurKernel;

    private OctreeGenerator octree;

    public GPULightingCalculator(OctreeGenerator octree)
    {
        context = Context.Create(builder => builder.Default().EnableAlgorithms());
        accelerator = context.GetPreferredDevice(preferCPU: false)
                             .CreateAccelerator(context);
        directKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Vector4>, ArrayView<Vector4>, ArrayView<float>, Vector3, int, float, float, float, float>(DirectLightingKernel);
        indirectKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Vector4>, ArrayView<Vector4>, ArrayView<float>, Vector3, int, float, float, float, float>(IndirectLightingKernel);
        blurKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Vector4>, ArrayView<Vector4>, ArrayView<float>, Vector3, int, float, float, float, float>(BlurKernel);

        this.octree = octree;
    }

    public void CalculateLighting(Vector3 lightDirection, int numBounces)
    {
        var posAndSize = octree.posAndSize;
        var nodeData = octree.nodeData;

        using (var devicePosAndSize = accelerator.Allocate1D<Vector4>(posAndSize))
        using (var deviceNodeData = accelerator.Allocate1D<Vector4>(nodeData))
        using (var deviceLightLevels = accelerator.Allocate1D<float>(octree.NodeCount))
        {
            devicePosAndSize.CopyFromCPU(posAndSize);
            deviceNodeData.CopyFromCPU(nodeData);

            // Launch the kernel
            directKernel(octree.NodeCount, devicePosAndSize.View, deviceNodeData.View, deviceLightLevels.View,lightDirection, numBounces, 64f, 1f, 0.3f, 0.15f);

            indirectKernel(octree.NodeCount, devicePosAndSize.View, deviceNodeData.View, deviceLightLevels.View, lightDirection, numBounces, 15f, 1f, 0.3f, 1f);
            indirectKernel(octree.NodeCount, devicePosAndSize.View, deviceNodeData.View, deviceLightLevels.View, lightDirection, numBounces, 15f, 1f, 0.3f, 1f);
            indirectKernel(octree.NodeCount, devicePosAndSize.View, deviceNodeData.View, deviceLightLevels.View, lightDirection, numBounces, 15f, 1f, 0.3f, 1f);

            blurKernel(octree.NodeCount, devicePosAndSize.View, deviceNodeData.View, deviceLightLevels.View, lightDirection, numBounces, 15f, 1f, 0.3f, 1f);
            blurKernel(octree.NodeCount, devicePosAndSize.View, deviceNodeData.View, deviceLightLevels.View, lightDirection, numBounces, 15f, 1f, 0.3f, 1f);
            blurKernel(octree.NodeCount, devicePosAndSize.View, deviceNodeData.View, deviceLightLevels.View, lightDirection, numBounces, 15f, 1f, 0.3f, 1f);
            blurKernel(octree.NodeCount, devicePosAndSize.View, deviceNodeData.View, deviceLightLevels.View, lightDirection, numBounces, 15f, 1f, 0.3f, 1f);
            // blurKernel(octree.NodeCount, devicePosAndSize.View, deviceNodeData.View, deviceLightLevels.View, lightDirection, numBounces, 15f, 1f, 0.3f, 1f);
            // blurKernel(octree.NodeCount, devicePosAndSize.View, deviceNodeData.View, deviceLightLevels.View, lightDirection, numBounces, 15f, 1f, 0.3f, 1f);
            // blurKernel(octree.NodeCount, devicePosAndSize.View, deviceNodeData.View, deviceLightLevels.View, lightDirection, numBounces, 15f, 1f, 0.3f, 1f);
            // blurKernel(octree.NodeCount, devicePosAndSize.View, deviceNodeData.View, deviceLightLevels.View, lightDirection, numBounces, 15f, 1f, 0.3f, 1f);


            // Copy results back to CPU
            deviceNodeData.CopyToCPU(nodeData);
        }
    }

    // A simple random number generator suitable for GPU use
    private static uint XorShift(ref uint state)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }

    // Generate a random float between 0 and 1
    private static float RandomFloat(ref uint state)
    {
        return (float)XorShift(ref state) / uint.MaxValue;
    }

    // Generate a random direction vector from a seed
    public static Vector3 GetRandomDirection(uint seed)
    {
        uint state = seed;

        // Generate random spherical coordinates
        float theta = RandomFloat(ref state) * 2 * XMath.PI;
        float phi = XMath.Acos(2 * RandomFloat(ref state) - 1);

        // Convert spherical coordinates to Cartesian coordinates
        float x = XMath.Sin(phi) * XMath.Cos(theta);
        float y = XMath.Sin(phi) * XMath.Sin(theta);
        float z = XMath.Cos(phi);

        return new Vector3(x, y, z);
    }

    private static void DirectLightingKernel(
        Index1D index, 
        ArrayView<Vector4> posAndSize, 
        ArrayView<Vector4> nodeData, 
        ArrayView<float> lightLevels,
        Vector3 lightDirection,
        int numBounces,
        float maxDistance,
        float stepSize,
        float directStrength,
        float indirectStrength)
    {
        Vector3 position = new Vector3(posAndSize[index].X, posAndSize[index].Y, posAndSize[index].Z);
        float size = posAndSize[index].W;
        Vector3 origin = position + new Vector3(size) * 1.1f;

        // Calculate direct lighting
        int hitIndex;
        if (CastRay(origin, lightDirection, maxDistance, stepSize, posAndSize, nodeData, out hitIndex))
        {
            nodeData[index].Z = 0;
        }
        else
        {
            nodeData[index].Z = directStrength;
        }
    }

    private static void IndirectLightingKernel(
        Index1D index, 
        ArrayView<Vector4> posAndSize, 
        ArrayView<Vector4> nodeData, 
        ArrayView<float> lightLevels,
        Vector3 lightDirection,
        int numBounces,
        float maxDistance,
        float stepSize,
        float directStrength,
        float indirectStrength)
    {
        Vector3 position = new Vector3(posAndSize[index].X, posAndSize[index].Y, posAndSize[index].Z);
        float size = posAndSize[index].W;
        Vector3 origin = position + new Vector3(size) * 1.1f;

        // Calculate indirect lighting
        float lightLevel = nodeData[index].Z;
        int hitIndex;

        for (int i = 0; i < 25; i++)
        {
            Vector3 direction = GetRandomDirection((uint)index * 71 + (uint)i * 182);

            if (CastRay(origin, direction, maxDistance, stepSize, posAndSize, nodeData, out hitIndex))
            {
                // add based on the light level of the hit node
                lightLevel += nodeData[hitIndex].Z / 10 / 10;
            }
        }

        // Store the new light level
        nodeData[index].Z = lightLevel;
    }

    private static void BlurKernel(
        Index1D index, 
        ArrayView<Vector4> posAndSize, 
        ArrayView<Vector4> nodeData, 
        ArrayView<float> lightLevels,
        Vector3 lightDirection,
        int numBounces,
        float maxDistance,
        float stepSize,
        float directStrength,
        float indirectStrength)
    {
        // average vlaues from surrounding nodes
        float total = 0;
        int count = 0;

        Vector3 position = new Vector3(posAndSize[index].X, posAndSize[index].Y, posAndSize[index].Z);
        float size = posAndSize[index].W;

        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                for (int k = -1; k <= 1; k++)
                {
                    Vector3 offset = new Vector3(i, j, k) * size;
                    Vector3 samplePos = position + offset;

                    int hitIndex;
                    if (CastRay(samplePos, lightDirection, 1, 1, posAndSize, nodeData, out hitIndex))
                    {
                        total += nodeData[hitIndex].Z;
                        count++;
                    }
                }
            }
        }

        if (count > 0)
        {
            nodeData[index].Z = total / count;
        }
    }

    private static bool CastRay(
        Vector3 origin, 
        Vector3 direction, 
        float maxDistance,
        float stepSize,
        ArrayView<Vector4> posAndSize,
        ArrayView<Vector4> nodeData,
        out int hitIndex)
    {
        float distance = 0;
        Vector3 currentPos = origin;

        while (distance < maxDistance)
        {
            int hitNodeIndex = FindNodeIndex(currentPos, posAndSize, nodeData);
            hitIndex = hitNodeIndex;
            if (hitNodeIndex < 0 || hitNodeIndex >= nodeData.Length)
            {
                return false;
            }

            var data = nodeData[hitNodeIndex];
            if (data.X != 0)
            {
                return true;
            }

            var size = posAndSize[hitNodeIndex].W;
            currentPos += direction * size * stepSize;
            distance += size * stepSize;
        }

        hitIndex = -1;
        return false;
    }

    private static int FindNodeIndex(Vector3 position, ArrayView<Vector4> posAndSize, ArrayView<Vector4> nodeData)
    {
        int currentIndex = 0;
        Vector4 rootNode = posAndSize[0];
        float size = rootNode.W;

        if (position.X < rootNode.X || position.Y < rootNode.Y || position.Z < rootNode.Z ||
            position.X >= rootNode.X + size || position.Y >= rootNode.Y + size || position.Z >= rootNode.Z + size)
        {
            return -1;
        }

        while (currentIndex < posAndSize.Length)
        {
            if (nodeData[currentIndex].Y == 1)
                return currentIndex;

            float childSize = size * 0.5f;

            int childOffset = ((position.X >= rootNode.X + childSize ? 1 : 0) |
                               (position.Y >= rootNode.Y + childSize ? 2 : 0) |
                               (position.Z >= rootNode.Z + childSize ? 4 : 0));

            currentIndex = currentIndex * 8 + 1 + childOffset;

            if ((childOffset & 1) != 0) rootNode.X += childSize;
            if ((childOffset & 2) != 0) rootNode.Y += childSize;
            if ((childOffset & 4) != 0) rootNode.Z += childSize;

            size = childSize;
        }

        return -1;
    }

    public void Dispose()
    {
        accelerator?.Dispose();
        context?.Dispose();
    }
}