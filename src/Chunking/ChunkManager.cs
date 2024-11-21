using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.InteropServices;
using Engine;
using Silk.NET.OpenGL;

namespace VoxelEngine;

public class ChunkManager : IDisposable
{
    public ConcurrentDictionary<Vector3, Chunk> ChunksLookup { get; init; }
    private readonly ConcurrentQueue<int> freeChunkIndices;
    public Chunk[] ChunksArray { get; private set; }
    private GlobalOctree globalOctree;

    public int ChunkSize { get; private set; }
    public float VoxelSize { get; private set; }
    private readonly int renderDistance;

    private uint positionsBuffer;
    private uint materialsBuffer;
    private uint lightLevelsBuffer;
    private uint chunkDataBuffer;

    private GLContext context;

    public const int MAX_CHUNKS = 64; 
    public int NodeCount { get; private set; }
    public int LeafCount { get; private set; }

    public unsafe ChunkManager(GLContext context, int chunkSize, float voxelSize, int renderDistance = 1)
    {
        TerrainGenerator.Initialize(chunkSize, voxelSize);
        this.context = context;
        this.ChunkSize = chunkSize;
        this.VoxelSize = voxelSize;
        this.renderDistance = renderDistance;
        this.ChunksLookup = new ConcurrentDictionary<Vector3, Chunk>();

        NodeCount = Chunk.CalculateNodeCount(chunkSize);
        LeafCount = Chunk.CalculateLeafCount(chunkSize);

        ChunksArray = new Chunk[MAX_CHUNKS];
        for (int i = 0; i < MAX_CHUNKS; i++)
        {
            ChunksArray[i] = new Chunk(Vector3.Zero, chunkSize, voxelSize);
        }

        freeChunkIndices = new ConcurrentQueue<int>();
        for (int i = 0; i < MAX_CHUNKS; i++)
        {
            freeChunkIndices.Enqueue(i);
        }

        globalOctree = new GlobalOctree(context, chunkSize);
        
        GenerateBuffers();
    }

    private unsafe void GenerateBuffers()
    {
        context.ExecuteCmd((dt, gl) =>
        {
            gl.GenBuffers(1, out positionsBuffer);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, positionsBuffer);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(sizeof(Vector4) * NodeCount * MAX_CHUNKS), null, BufferUsageARB.DynamicDraw);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);

            gl.GenBuffers(1, out materialsBuffer);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, materialsBuffer);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(sizeof(int) * LeafCount * MAX_CHUNKS), null, BufferUsageARB.DynamicDraw);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);

            gl.GenBuffers(1, out lightLevelsBuffer);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, lightLevelsBuffer);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(sizeof(Vector4) * LeafCount * MAX_CHUNKS), null, BufferUsageARB.DynamicDraw);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);

            gl.GenBuffers(1, out chunkDataBuffer);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, chunkDataBuffer);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(sizeof(ChunkData) * MAX_CHUNKS), null, BufferUsageARB.DynamicDraw);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);
        });

    }

    public void BindBuffers(uint? posLoc, uint? matLoc, uint? lightLoc, uint? colorLoc, uint? chunkDataLoc, uint? globalOctreeLoc)
    {
        context.ExecuteCmd((dt, gl) =>
        {
            if (posLoc != null) gl.BindBufferBase(GLEnum.ShaderStorageBuffer, posLoc.Value, positionsBuffer);
            if (matLoc != null) gl.BindBufferBase(GLEnum.ShaderStorageBuffer, matLoc.Value, materialsBuffer);
            if (lightLoc != null) gl.BindBufferBase(GLEnum.ShaderStorageBuffer, lightLoc.Value, lightLevelsBuffer);
            if (chunkDataLoc != null) gl.BindBufferBase(GLEnum.ShaderStorageBuffer, chunkDataLoc.Value, chunkDataBuffer);
            if (globalOctreeLoc != null) globalOctree.BindBuffer(globalOctreeLoc.Value);
        });

    }

    public Vector3 GetChunkPosition(Vector3 worldPosition)
    {
        float chunkWorldSize = ChunkSize * VoxelSize;
        return new Vector3(
            MathF.Floor(worldPosition.X / chunkWorldSize) * chunkWorldSize,
            MathF.Floor(worldPosition.Y / chunkWorldSize) * chunkWorldSize,
            MathF.Floor(worldPosition.Z / chunkWorldSize) * chunkWorldSize
        );
    }

    public void RebuildGlobalOctree()
    {
        globalOctree.Rebuild(ChunksLookup);
    }

    public void GenerateChunkTerrain(Vector3 atPosition)
    {
        var closestChunk = GetChunkPosition(atPosition);
        var success = freeChunkIndices.TryDequeue(out int dataIndex);
        if (!success)
        {
            Console.WriteLine($"No free chunk indices available for {closestChunk}");
            return;
        }

        var chunk = ChunksArray[dataIndex];

        if (!ChunksLookup.TryAdd(closestChunk, chunk))
        {
            Console.WriteLine($"Chunk already exists at {closestChunk}");
            return;
        }

        chunk.Move(closestChunk);
        chunk.BeginGenerate(dataIndex);

        Console.WriteLine($"Generating chunk at {closestChunk}");
    }

    public async Task ApplyChunks()
    {
        var finializedChunks = new List<Chunk>();
        var tasks = new List<Task>();
        foreach (var chunk in ChunksLookup.Values)
        {
            if (chunk.State == Chunk.ChunkState.AwaitFinalizedGeneration)
            {
                tasks.Add(Task.Run(() => chunk.FinalizeGeneration()));
                finializedChunks.Add(chunk);
            }
        }

        await Task.WhenAll(tasks);

        context.ExecuteCmd((dt, gl) =>
        {
            for (int i = 0; i < finializedChunks.Count; i++)
            {
                var chunk = finializedChunks[i];
                chunk.UploadAll(gl, chunk.ChunkIndex, positionsBuffer, materialsBuffer, lightLevelsBuffer, chunkDataBuffer);
            }
        });
    }

    public void Dispose()
    {
        context.ExecuteCmd((dt, gl) =>
        {
            gl.DeleteBuffer(positionsBuffer);
            gl.DeleteBuffer(materialsBuffer);
            gl.DeleteBuffer(lightLevelsBuffer);
            gl.DeleteBuffer(chunkDataBuffer);
        });

        globalOctree.Dispose();
    }
}