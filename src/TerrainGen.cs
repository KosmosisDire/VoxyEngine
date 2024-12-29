using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using Silk.NET.OpenGL;

namespace Engine
{
    public class TerrainGenerator
    {
        private const int BlockSize = 8;
        private const int ChunkSize = BlockSize * BlockSize;
        private const int GridSize = 5;
        private const float VoxelSize = 1.0f / ChunkSize;
        private const float NoiseScale = 0.3f;

        private readonly ConcurrentDictionary<int, uint[]> _chunkMasks = new();
        private readonly ConcurrentDictionary<int, uint[]> _blockMasks = new();
        private readonly ConcurrentDictionary<int, uint> _uncompressedMaterials = new();

        // Helper method to convert world space to normalized chunk space [0,1]
        private Vector3 WorldToNormalized(Vector3 worldPos)
        {
            return new Vector3(
                worldPos.X / ChunkSize,
                worldPos.Y / ChunkSize,
                worldPos.Z / ChunkSize
            );
        }

        private Vector3 GetSnappedPosition(Vector3 worldPos)
        {
            // Convert to normalized space first
            Vector3 normalizedPos = WorldToNormalized(worldPos);
            float snapSize = VoxelSize * 4f;
            
            // Perform snapping in normalized space
            return new Vector3(
                MathF.Floor(normalizedPos.X / snapSize) * snapSize,
                MathF.Floor(normalizedPos.Y / snapSize) * snapSize,
                MathF.Floor(normalizedPos.Z / snapSize) * snapSize
            ) * ChunkSize; // Convert back to world space
        }

        private bool GenerateSolid(Vector3 worldPos)
        {
            // Convert to normalized space
            Vector3 normalizedPos = WorldToNormalized(worldPos);
            
            // Calculate noise scale exactly as in GLSL
            float scale = NoiseScale * (EngineMath.SimplexNoise3D(
                normalizedPos.X * NoiseScale * 0.1f,
                normalizedPos.Y * NoiseScale * 0.1f,
                normalizedPos.Z * NoiseScale * 0.1f) + 1.0f);

            // Generate height map matching GLSL coordinate space
            float h = EngineMath.SimplexNoise2D(normalizedPos.X, normalizedPos.Z, scale * 0.1f) * 2.0f;
            h += EngineMath.SimplexNoise2D(normalizedPos.X, normalizedPos.Z, scale * 0.5f) * 0.5f;
            h += EngineMath.SimplexNoise2D(normalizedPos.X, normalizedPos.Z, scale * 0.9f) * 0.25f;
            
            // Scale and offset in chunk-space units, matching GLSL
            h = (h * 0.5f + 0.5f) * 2.0f + 1.0f;

            // Generate caves using normalized coordinates
            float cave = EngineMath.SimplexNoise3D(
                normalizedPos.X * scale,
                normalizedPos.Y * scale,
                normalizedPos.Z * scale);

            // Match GLSL comparison logic exactly
            return (normalizedPos.Y < h) && (cave > 0.0f);
        }

        private uint GetMaterialForPosition(Vector3 worldPos, bool solidAbove)
        {
            // Convert to normalized space
            Vector3 normalizedPos = WorldToNormalized(worldPos);
            
            // Calculate depth in normalized space to match GLSL
            float depth = 1.0f - normalizedPos.Y;

            float materialNoise = 
                EngineMath.SimplexNoise3D(normalizedPos.X, normalizedPos.Y, normalizedPos.Z, NoiseScale * 5.0f) * 0.7f +
                EngineMath.SimplexNoise3D(normalizedPos.X, normalizedPos.Y, normalizedPos.Z, NoiseScale * 10.0f) * 0.2f +
                EngineMath.SimplexNoise3D(normalizedPos.X, normalizedPos.Y, normalizedPos.Z, NoiseScale * 20.0f) * 0.1f;

            // Add high-frequency detail noise
            materialNoise += EngineMath.SimplexNoise3D(
                normalizedPos.X, normalizedPos.Y, normalizedPos.Z, NoiseScale * 100.0f) * 0.1f;

            // Material selection logic matching GLSL thresholds
            if (!solidAbove)
            {
                return materialNoise > 0.35f ? 9u : 2u; // Surface materials
            }
            
            if (depth < 0.2f || materialNoise > 0.55f)
            {
                return 1u; // Near-surface dirt
            }
            
            if (materialNoise < -0.55f)
            {
                return 4u; // Ore deposits
            }
            
            return 3u; // Stone
        }

        private int GetLocalIndex(Vector3 pos) =>
            (int)(pos.X + pos.Y * BlockSize + pos.Z * BlockSize * BlockSize);

        private Vector3 GetChunkPos(int index) =>
            new(index % GridSize, (index / GridSize) % GridSize, index / (GridSize * GridSize));

        private Vector3 ComposePosition(Vector3 chunkPos, Vector3 blockPos, Vector3 voxelPos)
        {
            // Match GLSL's component-wise division exactly
            return new Vector3(
                chunkPos.X + (blockPos.X / BlockSize) + (voxelPos.X / ChunkSize),
                chunkPos.Y + (blockPos.Y / BlockSize) + (voxelPos.Y / ChunkSize),
                chunkPos.Z + (blockPos.Z / BlockSize) + (voxelPos.Z / ChunkSize)
            );
        }

        private int GlobalIndex(int parentIndex, int localIndex) =>
            parentIndex * BlockSize * BlockSize * BlockSize + localIndex;

        private void AtomicSetChunkBit(int chunkIndex, int blockIndex)
        {
            uint baseIndex = (uint)chunkIndex << 2;
            uint arrayIndex = baseIndex + (uint)(blockIndex >> 7);
            uint uintIndex = ((uint)blockIndex >> 5) & 3u;
            uint bitIndex = (uint)blockIndex & 31u;
            uint bitMask = 1u << (int)bitIndex;

            _chunkMasks.AddOrUpdate((int)arrayIndex,
                _ => CreateBitArrayOf4(uintIndex, bitMask),
                (_, existing) =>
                {
                    existing[uintIndex] |= bitMask;
                    return existing;
                });
        }

        private void AtomicSetBlockBit(int blockIndex, int voxelIndex)
        {
            uint baseIndex = (uint)blockIndex << 2;
            uint arrayIndex = baseIndex + (uint)(voxelIndex >> 7);
            uint uintIndex = ((uint)voxelIndex >> 5) & 3u;
            uint bitIndex = (uint)voxelIndex & 31u;
            uint bitMask = 1u << (int)bitIndex;

            _blockMasks.AddOrUpdate((int)arrayIndex,
                _ => CreateBitArrayOf4(uintIndex, bitMask),
                (_, existing) =>
                {
                    existing[uintIndex] |= bitMask;
                    return existing;
                });
        }

        private static uint[] CreateBitArrayOf4(uint uintIndex, uint bitMask)
        {
            var arr = new uint[4];
            arr[uintIndex] = bitMask;
            return arr;
        }

        private void SetMaterial(int voxelIndex, uint materialIndex)
        {
            uint uintIndex = (uint)voxelIndex / 4;
            uint bitPosition = ((uint)voxelIndex % 4) * 8;
            uint setMask = (materialIndex & 0xFFu) << (int)bitPosition;
            uint clearMask = ~(0xFFu << (int)bitPosition);

            _uncompressedMaterials.AddOrUpdate((int)uintIndex,
                _ => setMask,
                (_, existing) => (existing & clearMask) | setMask
            );
        }

        public void GenerateAllTerrain(GL gl, uint chunkMasksBuffer, uint blockMasksBuffer, uint uncompressedMaterialsBuffer)
        {
            GenerateTerrainParallel(0, GridSize * GridSize * GridSize);
            UploadBufferData(gl, chunkMasksBuffer, blockMasksBuffer, uncompressedMaterialsBuffer);
        }

        public void GenerateTerrainParallel(int startChunkIndex, int chunkCount)
        {
            const int regionSize = 2;
            int totalChunks = GridSize * GridSize * GridSize;
            int endChunk = Math.Min(startChunkIndex + chunkCount, totalChunks);
            int chunksPerRegion = regionSize * regionSize * regionSize;
            int regionCount = (int)Math.Ceiling((double)(endChunk - startChunkIndex) / chunksPerRegion);

            Parallel.For(0, regionCount, regionIndex =>
            {
                for (int rx = 0; rx < regionSize; rx++)
                {
                    for (int ry = 0; ry < regionSize; ry++)
                    {
                        for (int rz = 0; rz < regionSize; rz++)
                        {
                            int cIndex = startChunkIndex +
                                       (regionIndex * chunksPerRegion) +
                                       (rz * regionSize * regionSize) + (ry * regionSize) + rx;
                            if (cIndex >= endChunk) return;
                            GenerateChunk(cIndex);
                        }
                    }
                }
            });
        }

        private void GenerateChunk(int chunkIndex)
        {
            Vector3 chunkPos = GetChunkPos(chunkIndex);

            for (int bx = 0; bx < BlockSize; bx++)
            {
                for (int by = 0; by < BlockSize; by++)
                {
                    for (int bz = 0; bz < BlockSize; bz++)
                    {
                        var blockPos = new Vector3(bx, by, bz);
                        var baseWorldPos = ComposePosition(chunkPos, blockPos, Vector3.Zero);
                        var worldBlockPos = GetSnappedPosition(baseWorldPos);

                        if (!GenerateSolid(worldBlockPos)) continue;

                        int blockLocalIndex = GetLocalIndex(blockPos);
                        AtomicSetChunkBit(chunkIndex, blockLocalIndex);

                        int globalBlockIndex = GlobalIndex(chunkIndex, blockLocalIndex);

                        for (int vx = 0; vx < BlockSize; vx++)
                        {
                            for (int vy = 0; vy < BlockSize; vy++)
                            {
                                for (int vz = 0; vz < BlockSize; vz++)
                                {
                                    var voxelPos = new Vector3(vx, vy, vz);
                                    int voxelLocalIndex = GetLocalIndex(voxelPos);
                                    AtomicSetBlockBit(globalBlockIndex, voxelLocalIndex);

                                    var worldPos = ComposePosition(chunkPos, blockPos, voxelPos);
                                    var worldPosAbove = worldBlockPos + new Vector3(0, VoxelSize * 4, 0);

                                    bool solidAbove = GenerateSolid(worldPosAbove);
                                    uint materialId = GetMaterialForPosition(worldPos, solidAbove);

                                    int globalVoxelIndex = GlobalIndex(globalBlockIndex, voxelLocalIndex);
                                    SetMaterial(globalVoxelIndex, materialId);
                                }
                            }
                        }
                    }
                }
            }
        }

        public unsafe void UploadBufferData(GL gl, uint chunkMasksBuffer, uint blockMasksBuffer, uint uncompressedMaterialsBuffer)
        {
            var sortedChunkMasks = _chunkMasks.OrderBy(kvp => kvp.Key).SelectMany(kvp => kvp.Value).ToArray();
            var sortedBlockMasks = _blockMasks.OrderBy(kvp => kvp.Key).SelectMany(kvp => kvp.Value).ToArray();
            var sortedMaterials = _uncompressedMaterials.OrderBy(kvp => kvp.Key).Select(kvp => kvp.Value).ToArray();

            fixed (void* chunkPtr = sortedChunkMasks)
            fixed (void* blockPtr = sortedBlockMasks)
            fixed (void* matsPtr = sortedMaterials)
            {
                gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, chunkMasksBuffer);
                gl.BufferSubData(BufferTargetARB.ShaderStorageBuffer, 0,
                    (nuint)(sortedChunkMasks.Length * sizeof(uint)), chunkPtr);

                gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, blockMasksBuffer);
                gl.BufferSubData(BufferTargetARB.ShaderStorageBuffer, 0,
                    (nuint)(sortedBlockMasks.Length * sizeof(uint)), blockPtr);

                gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, uncompressedMaterialsBuffer);
                gl.BufferSubData(BufferTargetARB.ShaderStorageBuffer, 0,
                    (nuint)(sortedMaterials.Length * sizeof(uint)), matsPtr);

                gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);
            }
        }
    }
}