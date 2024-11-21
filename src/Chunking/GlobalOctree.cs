using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.InteropServices;
using Engine;
using Silk.NET.OpenGL;

[StructLayout(LayoutKind.Sequential)]
public struct GlobalOctreeNode
{
    public Vector3 position;
    public float size;
    public int chunkIndex;  // Index into chunk array, -1 if not a leaf or empty
    public int padding1;
    public int padding2;
    public int padding3;
}

public class GlobalOctree : IDisposable
{
    private GlobalOctreeNode[] nodes;
    private uint nodeBuffer;
    private GLContext context;
    private float chunkSize;

    public int NodeCount = 0;

    public unsafe GlobalOctree(GLContext context, float chunkSize)
    {
        this.context = context;
        this.chunkSize = chunkSize;
        nodes = [];
        
        context.ExecuteCmd((dt, gl) =>
        {
            gl.GenBuffers(1, out nodeBuffer);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, nodeBuffer);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(sizeof(GlobalOctreeNode) * 1), null, BufferUsageARB.StaticDraw);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);
        });
    }

    public unsafe void Upload()
    {
        context.ExecuteCmd((dt, gl) =>
        {
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, nodeBuffer);
            nuint bufferSize = (nuint)(sizeof(GlobalOctreeNode) * NodeCount);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, bufferSize, [..nodes], BufferUsageARB.StaticDraw);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);
        });
    }

    public void BindBuffer(uint binding)
    {
        context.ExecuteCmd((dt, gl) =>
        {
            gl.BindBufferBase(GLEnum.ShaderStorageBuffer, binding, nodeBuffer);
        });
    }

    public void Rebuild(ConcurrentDictionary<Vector3, Chunk> chunks)
    {
        if (chunks.Count == 0)
        {
            Upload();
            return;
        }

        Vector3 min = new Vector3(float.MaxValue);
        Vector3 max = new Vector3(float.MinValue);

        foreach (var chunk in chunks.Values)
        {
            min = Vector3.Min(min, chunk.Position);
            max = Vector3.Max(max, chunk.Position + new Vector3(chunkSize));
        }

        // Calculate size needed to encompass all chunks (next power of 2)
        float size = Math.Max(Math.Max(max.X - min.X, max.Y - min.Y), max.Z - min.Z);
        size = MathF.Pow(2, MathF.Ceiling(MathF.Log2(size)));

        // Calculate number of nodes needed
        NodeCount = Chunk.CalculateNodeCount((int)(size / chunkSize));
        nodes = new GlobalOctreeNode[NodeCount];

        // Build octree recursively
        BuildNode(min, size, 0, chunks);

        Upload();
    }

    private static readonly Vector3[] childOffsets = [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
        new Vector3(1, 1, 0),
        new Vector3(0, 0, 1),
        new Vector3(1, 0, 1),
        new Vector3(0, 1, 1),
        new Vector3(1, 1, 1)
    ];

    private bool BuildNode(Vector3 position, float size, int index, ConcurrentDictionary<Vector3, Chunk> chunks)
    {
        nodes[index] = new GlobalOctreeNode
        {
            position = position,
            size = size,
            chunkIndex = -1
        };

        float childSize = size / 2f;

        if (childSize < chunkSize)
        {
            if (chunks.TryGetValue(nodes[index].position, out var chunk))
            {
                nodes[index].chunkIndex = chunk.ChunkIndex;
            }

            nodes[index].size = -size;  // Mark as leaf node
            return true;
        }

        bool allEmpty = true;
        for (int i = 0; i < 8; i++)
        {
            Vector3 offset = childOffsets[i] * childSize;
            Vector3 childPosition = position + offset;
            int childIndex = index * 8 + 1 + i;
            var isChunk = BuildNode(childPosition, childSize, childIndex, chunks);

            if (isChunk || nodes[childIndex].size > 0)
            {
                allEmpty = false;
            }
        }

        if (allEmpty)
        {
            nodes[index].size = -size;  // Mark as empty node
        }

        return false;
    }

    public void Dispose()
    {
        context.ExecuteCmd((dt, gl) =>
        {
            gl.DeleteBuffer(nodeBuffer);
        });
    }
}