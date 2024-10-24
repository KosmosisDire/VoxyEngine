using System.Collections.Concurrent;
using System.Numerics;
using Engine;
using Silk.NET.OpenGL;

namespace VoxelEngine;

public class Chunk
{
    public Vector3 Position { get; private set; }
    public bool IsLoaded { get; set; }
    public bool IsGenerating { get; set; }
    public Vector4[] Positions { get; set; }
    public int[] Materials { get; set; }
    public Vector4[] LightLevels { get; set; }
    public Vector4[] Colors { get; set; }
    public int NodeCount { get; set; }
    public int LeafCount { get; set; }

    public Chunk(Vector3 position)
    {
        Position = position;
        IsLoaded = false;
        IsGenerating = false;
    }
}

public class ChunkManager : IDisposable
{
    private readonly ConcurrentDictionary<Vector3, Chunk> chunks;
    private readonly TerrainGenerator terrainGenerator;
    private readonly int chunkSize;
    private readonly float voxelSize;
    private readonly int renderDistance;
    private readonly object bufferLock = new object();

    // GPU buffer management
    private uint[] positionsBuffers;
    private uint[] materialsBuffers;
    private uint[] lightLevelsBuffers;
    private uint[] colorsBuffers;
    private GLContext context;

    public ChunkManager(GLContext context, int chunkSize, float voxelSize, int renderDistance = 8)
    {
        this.context = context;
        this.chunkSize = chunkSize;
        this.voxelSize = voxelSize;
        this.renderDistance = renderDistance;
        this.chunks = new ConcurrentDictionary<Vector3, Chunk>();
        this.terrainGenerator = new TerrainGenerator();

        // Initialize buffer arrays
        positionsBuffers = new uint[renderDistance * renderDistance * renderDistance];
        materialsBuffers = new uint[renderDistance * renderDistance * renderDistance];
        lightLevelsBuffers = new uint[renderDistance * renderDistance * renderDistance];
        colorsBuffers = new uint[renderDistance * renderDistance * renderDistance];
    }

    public Vector3 GetChunkPosition(Vector3 worldPosition)
    {
        return new Vector3(
            MathF.Floor(worldPosition.X / (chunkSize * voxelSize)) * chunkSize * voxelSize,
            MathF.Floor(worldPosition.Y / (chunkSize * voxelSize)) * chunkSize * voxelSize,
            MathF.Floor(worldPosition.Z / (chunkSize * voxelSize)) * chunkSize * voxelSize
        );
    }

    public void UpdateChunks(Vector3 cameraPosition)
    {
        Vector3 centerChunkPos = GetChunkPosition(cameraPosition);
        float chunkWorldSize = chunkSize * voxelSize;
        HashSet<Vector3> chunksToKeep = new HashSet<Vector3>();

        // Determine which chunks should be loaded
        for (int x = -renderDistance; x <= renderDistance; x++)
        {
            for (int y = -renderDistance; y <= renderDistance; y++)
            {
                for (int z = -renderDistance; z <= renderDistance; z++)
                {
                    Vector3 offset = new Vector3(x, y, z) * chunkWorldSize;
                    Vector3 chunkPos = centerChunkPos + offset;

                    if (Vector3.Distance(chunkPos + new Vector3(chunkWorldSize * 0.5f), cameraPosition) <= renderDistance * chunkWorldSize)
                    {
                        chunksToKeep.Add(chunkPos);

                        if (!chunks.ContainsKey(chunkPos))
                        {
                            var chunk = new Chunk(chunkPos);
                            if (chunks.TryAdd(chunkPos, chunk))
                            {
                                GenerateChunk(chunk);
                            }
                        }
                    }
                }
            }
        }

        // Unload chunks that are too far
        foreach (var chunk in chunks)
        {
            if (!chunksToKeep.Contains(chunk.Key))
            {
                if (chunks.TryRemove(chunk.Key, out var removedChunk))
                {
                    UnloadChunk(removedChunk);
                }
            }
        }
    }

    private unsafe void GenerateChunk(Chunk chunk)
    {
        if (chunk.IsGenerating) return;
        chunk.IsGenerating = true;

        Task.Run(() =>
        {
            var octreeGenerator = new OctreeGenerator(chunkSize);
            octreeGenerator.GenerateOctree(chunk.Position);

            chunk.NodeCount = octreeGenerator.NodeCount;
            chunk.LeafCount = octreeGenerator.LeafCount;
            chunk.Positions = octreeGenerator.positions;
            chunk.Materials = octreeGenerator.materials;
            chunk.LightLevels = octreeGenerator.lightLevels;
            chunk.Colors = octreeGenerator.colors;

            context.ExecuteCmd((dt, gl) =>
            {
                lock (bufferLock)
                {
                    // Create and upload buffers for the chunk
                    var buffers = new uint[4];
                    gl.GenBuffers(4, buffers);

                    // Store buffer IDs in arrays using chunk position as key
                    int index = GetBufferIndex(chunk.Position);
                    positionsBuffers[index] = buffers[0];
                    materialsBuffers[index] = buffers[1];
                    lightLevelsBuffers[index] = buffers[2];
                    colorsBuffers[index] = buffers[3];

                    // Upload data to buffers
                    gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, buffers[0]);
                    gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(chunk.NodeCount * sizeof(float) * 4), (ReadOnlySpan<Vector4>)chunk.Positions, BufferUsageARB.StaticDraw);

                    gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, buffers[1]);
                    gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(chunk.LeafCount * sizeof(int)), (ReadOnlySpan<int>)chunk.Materials, BufferUsageARB.DynamicDraw);

                    gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, buffers[2]);
                    gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(chunk.LeafCount * sizeof(float) * 4), (ReadOnlySpan<Vector4>)chunk.LightLevels, BufferUsageARB.DynamicDraw);

                    gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, buffers[3]);
                    gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(chunk.LeafCount * sizeof(float) * 4), (ReadOnlySpan<Vector4>)chunk.Colors, BufferUsageARB.DynamicDraw);

                    gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);
                }

                chunk.IsLoaded = true;
                chunk.IsGenerating = false;
            });
        });
    }

    private void UnloadChunk(Chunk chunk)
    {
        if (!chunk.IsLoaded) return;

        context.ExecuteCmd((dt, gl) =>
        {
            lock (bufferLock)
            {
                int index = GetBufferIndex(chunk.Position);

                // Delete buffers
                gl.DeleteBuffer(positionsBuffers[index]);
                gl.DeleteBuffer(materialsBuffers[index]);
                gl.DeleteBuffer(lightLevelsBuffers[index]);
                gl.DeleteBuffer(colorsBuffers[index]);

                // Clear buffer IDs
                positionsBuffers[index] = 0;
                materialsBuffers[index] = 0;
                lightLevelsBuffers[index] = 0;
                colorsBuffers[index] = 0;
            }
        });

        chunk.IsLoaded = false;
    }

    private int GetBufferIndex(Vector3 chunkPosition)
    {
        // Convert chunk position to local coordinates within render distance
        Vector3 localPos = chunkPosition / (chunkSize * voxelSize) + new Vector3(renderDistance);
        localPos = new Vector3(localPos.X % renderDistance * 2, localPos.Y % renderDistance * 2, localPos.Z % renderDistance * 2);
        return (int)(localPos.X + localPos.Y * renderDistance + localPos.Z * renderDistance * renderDistance);
    }

    public IEnumerable<(Chunk chunk, uint posBuffer, uint matBuffer, uint lightBuffer, uint colorBuffer)> GetVisibleChunks()
    {
        foreach (var chunk in chunks.Values.Where(c => c.IsLoaded))
        {
            int index = GetBufferIndex(chunk.Position);
            yield return (
                chunk,
                positionsBuffers[index],
                materialsBuffers[index],
                lightLevelsBuffers[index],
                colorsBuffers[index]
            );
        }
    }

    public void Dispose()
    {
        foreach (var chunk in chunks.Values)
        {
            UnloadChunk(chunk);
        }
    }
}