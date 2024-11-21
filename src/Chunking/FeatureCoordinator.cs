using System.Collections.Concurrent;
using System.Numerics;

public class FeatureCoordinator
{
    private class PendingFeature
    {
        public ITerrainFeature Feature { get; set; }
        public Vector3 WorldPosition { get; set; }
        public HashSet<Vector3> RequiredChunks { get; set; }
        public Dictionary<Vector3, (int, Vector4)> Blocks { get; set; }
    }

    private readonly ConcurrentDictionary<Vector3, HashSet<PendingFeature>> pendingFeaturesByChunk = new();
    private readonly object coordinationLock = new();
    private readonly int chunkSize;
    private readonly float voxelSize;

    public FeatureCoordinator(int chunkSize, float voxelSize)
    {
        this.chunkSize = chunkSize;
        this.voxelSize = voxelSize;
    }

    public void RegisterFeature(Vector3 worldPos, ITerrainFeature feature, Vector3 bounds)
    {
        var affectedChunks = GetAffectedChunks(worldPos, bounds);
        var pendingFeature = new PendingFeature
        {
            Feature = feature,
            WorldPosition = worldPos,
            RequiredChunks = affectedChunks,
            Blocks = new Dictionary<Vector3, (int, Vector4)>()
        };

        lock (coordinationLock)
        {
            foreach (var chunkPos in affectedChunks)
            {
                pendingFeaturesByChunk.AddOrUpdate(
                    chunkPos,
                    new HashSet<PendingFeature> { pendingFeature },
                    (_, set) => { set.Add(pendingFeature); return set; }
                );
            }
        }
    }

    public void ProcessChunkFeatures(Vector3 chunkPosition, ConcurrentDictionary<Vector3, (int, Vector4)> chunkBlocks)
    {
        if (!pendingFeaturesByChunk.TryGetValue(chunkPosition, out var pendingFeatures))
        {
            return;
        }

        lock (coordinationLock)
        {
            foreach (var feature in pendingFeatures)
            {
                // Convert world position to local chunk position
                Vector3 localPos = WorldToLocalPosition(feature.WorldPosition, chunkPosition);
                
                // Generate the feature's blocks
                if (feature.Feature.ShouldGenerate(feature.WorldPosition))
                {
                    feature.Feature.Generate(localPos, feature.WorldPosition, feature.Blocks);
                }

                // Copy blocks that belong to this chunk
                foreach (var (pos, blockData) in feature.Blocks)
                {
                    Vector3 worldPos = LocalToWorldPosition(pos, chunkPosition);
                    if (GetChunkPosition(worldPos) == chunkPosition)
                    {
                        chunkBlocks.AddOrUpdate(
                            pos,
                            blockData,
                            (_, _) => blockData
                        );
                    }
                }

                // Remove this chunk from the feature's requirements
                feature.RequiredChunks.Remove(chunkPosition);

                // If all required chunks have been processed, remove the feature
                if (feature.RequiredChunks.Count == 0)
                {
                    pendingFeatures.Remove(feature);
                }
            }

            // If no more pending features for this chunk, remove the entry
            if (pendingFeatures.Count == 0)
            {
                pendingFeaturesByChunk.TryRemove(chunkPosition, out _);
            }
        }
    }

    private HashSet<Vector3> GetAffectedChunks(Vector3 worldPos, Vector3 bounds)
    {
        var chunks = new HashSet<Vector3>();
        float chunkWorldSize = chunkSize * voxelSize;
        
        // Calculate the world space bounds of the feature
        Vector3 halfBounds = bounds * voxelSize * 0.5f;
        Vector3 min = worldPos - halfBounds;
        Vector3 max = worldPos + halfBounds;

        // Get all chunk positions that intersect with the feature bounds
        for (float x = min.X; x <= max.X; x += chunkWorldSize)
        for (float y = min.Y; y <= max.Y; y += chunkWorldSize)
        for (float z = min.Z; z <= max.Z; z += chunkWorldSize)
        {
            chunks.Add(GetChunkPosition(new Vector3(x, y, z)));
        }

        return chunks;
    }

    private Vector3 GetChunkPosition(Vector3 worldPosition)
    {
        float chunkWorldSize = chunkSize * voxelSize;
        return new Vector3(
            MathF.Floor(worldPosition.X / chunkWorldSize) * chunkWorldSize,
            MathF.Floor(worldPosition.Y / chunkWorldSize) * chunkWorldSize,
            MathF.Floor(worldPosition.Z / chunkWorldSize) * chunkWorldSize
        );
    }

    private Vector3 WorldToLocalPosition(Vector3 worldPos, Vector3 chunkPos)
    {
        return (worldPos - chunkPos) / voxelSize;
    }

    private Vector3 LocalToWorldPosition(Vector3 localPos, Vector3 chunkPos)
    {
        return localPos * voxelSize + chunkPos;
    }
}