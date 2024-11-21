using System.Numerics;
using System.Collections.Immutable;
using Engine;
using System.Collections.Concurrent;


public static class TerrainGenerator
{
    public static ImmutableArray<Vector4> Palette = 
    [
        // neutral
        new Vector4(1.0f, 1.0f, 1.0f, 1.0f),  // White
        new Vector4(0.0f, 0.0f, 0.0f, 1.0f),  // Black
        new Vector4(0.5f, 0.5f, 0.5f, 1.0f),  // Gray

        // Earth tones
        new Vector4(0.627f, 0.322f, 0.176f, 1.0f),  // Brown
        new Vector4(0.824f, 0.706f, 0.549f, 1.0f),  // Tan
        new Vector4(0.565f, 0.490f, 0.373f, 1.0f),  // Taupe
        
        // Forest colors
        new Vector4(0.133f, 0.545f, 0.133f, 1.0f),  // Forest Green
        new Vector4(0.180f, 0.404f, 0.259f, 1.0f),  // Dark Green
        new Vector4(0.3f, 0.38f, 0.37f, 1.0f),  // Sage
        
        // Sky and Water
        new Vector4(0.529f, 0.808f, 0.922f, 1.0f),  // Sky Blue
        new Vector4(0.125f, 0.698f, 0.667f, 1.0f),  // Turquoise
        new Vector4(0.251f, 0.878f, 0.816f, 1.0f),  // Aqua
        
        // Stone
        new Vector4(0.663f, 0.663f, 0.663f, 1.0f),  // Gray
        new Vector4(0.804f, 0.784f, 0.745f, 1.0f),  // Beige
        new Vector4(0.541f, 0.520f, 0.486f, 1.0f),  // Warm Gray
        
        // Plant Life
        new Vector4(0.678f, 0.847f, 0.902f, 1.0f),  // Light Blue
        new Vector4(0.941f, 0.902f, 0.549f, 1.0f),  // Light Yellow
        new Vector4(0.196f, 0.804f, 0.196f, 1.0f),  // Lime Green
        
        // Sand and Desert
        new Vector4(0.957f, 0.643f, 0.376f, 1.0f),  // Sandy Orange
        new Vector4(0.824f, 0.706f, 0.549f, 1.0f),  // Desert Sand
        new Vector4(0.871f, 0.722f, 0.529f, 1.0f),   // Khaki
    ];

    public const float NOISE_SCALE = 0.01f;
    private const int MAX_HEIGHT = 64;
    private const int WATER_LEVEL = -32;
    private const int MIN_HEIGHT = -64;

    private static Vector2 noiseOffset;
    private static readonly List<IBiome> biomes;

    private static FeatureCoordinator featureCoordinator;



    static TerrainGenerator()
    {
        StaticRandom.Seed = 0;
        biomes = new List<IBiome>
        {
            new MountainBiome(),
            new ForestBiome(),
        };
    }

    public static void Initialize(int chunkSize, float voxelSize)
    {
        featureCoordinator = new FeatureCoordinator(chunkSize, voxelSize);
    }

    public static float GetNoise(Vector3 position)
    {
        return Perlin.Noise(
            position.X + noiseOffset.X,
            position.Y,
            position.Z + noiseOffset.Y
        );
    }

    public static float GetFlatNoise(Vector3 position)
    {
        return Perlin.Noise(
            position.X + noiseOffset.X,
            0,
            position.Z + noiseOffset.Y
        );
    }

    private static (IBiome, float)[] GetBiomeWeights(Vector3 position)
    {
        var weights = biomes
            .Select(b => (biome: b, weight: b.GetPriority(position)))
            .OrderByDescending(x => x.weight)
            .Take(2)
            .ToArray();

        // Normalize weights
        float total = weights.Sum(w => w.weight);
        return weights.Select(w => (w.biome, w.weight / total)).ToArray();
    }

    private static float GetSurfaceHeight(Vector3 position)
    {
        float baseHeight = GetBaseTerrainNoise(position);
        var weights = GetBiomeWeights(position);
        
        // Blend heights from different biomes
        return weights.Sum(w => w.Item2 * w.Item1.GetSurfaceHeight(position, baseHeight));
    }

    private static float GetBaseTerrainNoise(Vector3 position)
    {
        float noise = 0f;
        float amplitude = 1f;
        float frequency = 1f;
        float totalAmplitude = 0f;

        for (int i = 0; i < 3; i++)
        {
            noise += amplitude * GetNoise(new Vector3(
                position.X * NOISE_SCALE * frequency,
                0,
                position.Z * NOISE_SCALE * frequency
            ));
            totalAmplitude += amplitude;
            amplitude *= 0.5f;
            frequency *= 2f;
        }

        return noise / totalAmplitude;
    }

    public static void GenerateChunk(Vector3 chunkPosition, int chunkSize, float voxelSize, ConcurrentDictionary<Vector3, (int, Vector4)> blocks)
    {
        // generate simple perlin
        for (int x = 0; x < chunkSize; x++)
        for (int z = 0; z < chunkSize; z++)
        for (int y = 0; y < chunkSize; y++)
        {
            var noise = Perlin.Noise(
                chunkPosition.X * 0.01f + x * 0.01f,
                chunkPosition.Y * 0.01f + y * 0.01f,
                chunkPosition.Z * 0.01f + z * 0.01f
            );
            blocks[new Vector3(x, y, z)] = noise > 0 ? (1, Palette[0]) : (0, Palette[1]);
        }
        return;
        // Generate base terrain
        for (int x = 0; x < chunkSize; x++)
        for (int z = 0; z < chunkSize; z++)
        {
            // Calculate world position for noise sampling
            Vector3 worldPos = new(
                chunkPosition.X + x * voxelSize,
                0,
                chunkPosition.Z + z * voxelSize
            );
            
            float surfaceHeight = GetSurfaceHeight(worldPos);
            int height = MIN_HEIGHT + (int)((MAX_HEIGHT - MIN_HEIGHT) * (surfaceHeight + 1) * 0.5f);
            height = Math.Clamp(height, MIN_HEIGHT, MAX_HEIGHT);

            // Calculate local position for block placement
            Vector3 localColumnPos = new(
                x,
                0,
                z
            );

            for (int y = 0; y < chunkSize; y++)
            {
                Vector3 localBlockPos = localColumnPos + new Vector3(0, y, 0);
                Vector3 worldBlockPos = worldPos + new Vector3(0, chunkPosition.Y + y * voxelSize, 0);

                if (worldBlockPos.Y < height)
                {
                    // Get block type from biome blend
                    float heightRatio = (float)y / MAX_HEIGHT;
                    var weights = GetBiomeWeights(worldBlockPos);
                    var block = weights[0].Item1.GetBlockType(worldBlockPos, heightRatio);
                    blocks[localBlockPos] = block;
                }
                else if (worldBlockPos.Y < WATER_LEVEL)
                {
                    blocks[localBlockPos] = (1, Palette[11]); // Water
                }
            }

            // Generate surface features
            var worldSurfacePos = worldPos + new Vector3(0, chunkPosition.Y + height * voxelSize, 0);
            if (worldSurfacePos.Y > WATER_LEVEL)
            {
                var weights = GetBiomeWeights(worldSurfacePos);
                foreach (var (biome, weight) in weights)
                {
                    if (StaticRandom.NextDouble() < weight)
                    {
                        foreach (var feature in biome.GetFeatures(worldSurfacePos))
                        {
                            if (feature.ShouldGenerate(worldSurfacePos))
                            {
                                featureCoordinator.RegisterFeature(worldSurfacePos, feature, feature.GetBounds());
                            }
                        }
                    }
                }
            }
        }
    }

    public static void ApplyFeatures(Vector3 chunkPosition, ConcurrentDictionary<Vector3, (int, Vector4)> blocks)
    {
        // Process any features that affect this chunk
        featureCoordinator.ProcessChunkFeatures(chunkPosition, blocks);
    }

}