using System.Collections.Immutable;
using System.Numerics;
using Engine;

// Abstract base class for surface features
public abstract class SurfaceFeature
{
    protected readonly TerrainGenerator terrainGenerator;
    protected readonly Random random;

    protected SurfaceFeature(TerrainGenerator generator, Random random)
    {
        this.terrainGenerator = generator;
        this.random = random;
    }

    public abstract void Generate(Vector3 position, Dictionary<Vector3, (int, Vector4)> blocks);
}

public class PineTree : SurfaceFeature
{
    private const int MIN_HEIGHT = 6;
    private const int MAX_HEIGHT = 12;
    private const int LEAF_LAYERS = 5;

    public PineTree(TerrainGenerator generator, Random random) : base(generator, random) { }

    public override void Generate(Vector3 position, Dictionary<Vector3, (int, Vector4)> blocks)
    {
        int height = random.Next(MIN_HEIGHT, MAX_HEIGHT + 1);
        
        // Generate trunk
        for (int y = 0; y < height; y++)
        {
            Vector3 trunkPos = new(position.X, position.Y + y, position.Z);
            blocks[trunkPos] = (1, terrainGenerator.WoodColor);
        }

        // Generate pine leaves in a cone shape
        for (int layer = 0; layer < LEAF_LAYERS; layer++)
        {
            int layerY = height - (LEAF_LAYERS - layer);
            int radius = Math.Max(1, (LEAF_LAYERS - layer) / 2);

            for (int x = -radius; x <= radius; x++)
            {
                for (int z = -radius; z <= radius; z++)
                {
                    // Create cone shape by checking distance from center
                    if (Math.Sqrt(x * x + z * z) <= radius)
                    {
                        Vector3 leafPos = new(
                            position.X + x,
                            position.Y + layerY,
                            position.Z + z
                        );
                        
                        // Only place leaves where there isn't already a trunk
                        if (!blocks.ContainsKey(leafPos))
                        {
                            blocks[leafPos] = (1, terrainGenerator.LeafColor);
                        }
                    }
                }
            }
        }
    }
}

public class TerrainGenerator
{
    private ImmutableArray<Vector4> palette = 
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

    private readonly Random globalRandom;
    private const float NOISE_SCALE = 0.01f;
    private const float BIOME_SCALE = 0.01f;
    private const float CAVE_SCALE = 0.05f;
    private const float CAVE_THRESHOLD = 0.7f;
    private const int MAX_HEIGHT = 128;
    private const int WATER_LEVEL = 32;
    private const int MIN_HEIGHT = 28;
    
    // Biome thresholds
    private const float MOUNTAIN_THRESHOLD = 0.7f;
    private const float PLAINS_THRESHOLD = 0.4f;
    private const float DESERT_THRESHOLD = 0.2f;
// Add new colors for tree features
    public Vector4 WoodColor => palette[3];  // Brown
    public Vector4 LeafColor => palette[7];  // Dark Green

    private readonly Dictionary<string, SurfaceFeature> features;
    private const float TREE_SPACING = 0.2f;
    private const float TREE_NOISE_SCALE = 0.05f;

    private Vector2 noiseOffset = new(0, 0);

    public TerrainGenerator(int seed = 0)
    {
        globalRandom = new Random(seed);
        features = new Dictionary<string, SurfaceFeature>
        {
            { "pine_tree", new PineTree(this, globalRandom) }
        };

        // Randomize noise offset
        noiseOffset = new Vector2((globalRandom.NextSingle() - 0.5f) * 10000, (globalRandom.NextSingle()-0.5f) * 10000);
    }

    public float GetNoise(Vector3 position)
    {
        return Perlin.Noise(
            position.X + noiseOffset.X,
            position.Y,
            position.Z + noiseOffset.Y
        );
    }

    public float GetNoise(float x, float y, float z)
    {
        return Perlin.Noise(
            x + noiseOffset.X,
            y,
            z + noiseOffset.Y
        );
    }

    public float GetFlatNoise(Vector3 position)
    {
        return Perlin.Noise(
            position.X + noiseOffset.X,
            0,
            position.Z + noiseOffset.Y
        );
    }   

    public Dictionary<Vector3, (int, Vector4)> GenerateChunkWithFeatures(Vector3 chunkPosition, float chunkWidth)
    {
        var blocks = new Dictionary<Vector3, (int, Vector4)>();
        
        // Generate base terrain first
        for (int x = 0; x < chunkWidth; x++)
        {
            for (int z = 0; z < chunkWidth; z++)
            {
                for (int y = 0; y < MAX_HEIGHT; y++)
                {
                    Vector3 worldPos = new(
                        chunkPosition.X + x,
                        y,
                        chunkPosition.Z + z
                    );
                    
                    var block = GetTerrainAt(worldPos, chunkWidth);
                    if (block.Item1 != 0)  // If not air
                    {
                        blocks[worldPos] = block;
                    }
                }
            }
        }

        // Add surface features
        GenerateSurfaceFeatures(chunkPosition, chunkWidth, blocks);
        
        return blocks;
    }

    private void GenerateSurfaceFeatures(Vector3 chunkPosition, float chunkWidth, Dictionary<Vector3, (int, Vector4)> blocks)
    {
        for (int x = 0; x < chunkWidth; x++)
        {
            for (int z = 0; z < chunkWidth; z++)
            {
                Vector3 worldPos = new(
                    chunkPosition.X + x,
                    0,
                    chunkPosition.Z + z
                );

                float biomeNoise = GetFlatNoise(worldPos * BIOME_SCALE);
                
                // Only generate trees in forest biome
                if (biomeNoise <= DESERT_THRESHOLD)
                {
                    float treeNoise = GetFlatNoise(worldPos * TREE_NOISE_SCALE);

                    if (treeNoise > 1 - TREE_SPACING)
                    {
                        // Find surface height
                        int surfaceY = GetSurfaceHeight(worldPos);
                        
                        // Only place trees if we're above water level and have enough space
                        if (surfaceY > WATER_LEVEL && surfaceY + MAX_HEIGHT < MAX_HEIGHT)
                        {
                            Vector3 treePos = new(worldPos.X, surfaceY, worldPos.Z);
                            features["pine_tree"].Generate(treePos, blocks);
                        }
                    }
                }
            }
        }
    }


    public (int, Vector4) GetTerrainAt(Vector3 position, float chunkWidth)
    {
        // Get the surface height at this x,z coordinate
        int surfaceHeight = GetSurfaceHeight(position);
        
        // If we're below ground level
        if (position.Y < surfaceHeight)
        {
            // Cave generation
            if (position.Y < surfaceHeight - 5 && IsCaveLocation(position))
            {
                return GenerateCaveFeatures(position);
            }
            
            // Underground layers
            if (position.Y < surfaceHeight - 4)
            {
                return (1, palette[12]); // Stone
            }
            
            // Get biome for surface layer generation
            float biomeNoise = GetFlatNoise(position * BIOME_SCALE);
            return GetBiomeFeatures(position, (float)position.Y / MAX_HEIGHT, biomeNoise, chunkWidth);
        }
        
        // Water generation
        if (position.Y < WATER_LEVEL)
        {
            return (1, palette[11]); // Aqua for water
        }

        // Air above surface
        return (0, palette[0]);
    }

    private int GetSurfaceHeight(Vector3 position)
    {
        // Base terrain height
        float heightNoise = GetBaseTerrainNoise(new Vector3(position.X, 0, position.Z));
        
        // Get biome influence
        float biomeNoise = GetFlatNoise(position * BIOME_SCALE);
        
        // Adjust height based on biome
        float biomeHeight;
        if (biomeNoise > MOUNTAIN_THRESHOLD)
        {
            biomeHeight = heightNoise * 1.5f; // Mountains are taller
        }
        else if (biomeNoise > PLAINS_THRESHOLD)
        {
            biomeHeight = heightNoise * 0.7f; // Plains are flatter
        }
        else if (biomeNoise > DESERT_THRESHOLD)
        {
            biomeHeight = heightNoise * 0.8f; // Desert has medium variation
        }
        else
        {
            biomeHeight = heightNoise * 0.9f; // Forest has slight variation
        }
        
        // Convert to actual height value
        int height = MIN_HEIGHT + (int)((MAX_HEIGHT - MIN_HEIGHT) * (biomeHeight + 1) * 0.5f);
        return Math.Clamp(height, MIN_HEIGHT, MAX_HEIGHT);
    }

    private float GetBaseTerrainNoise(Vector3 position)
    {
        float noise = 0f;
        float amplitude = 1f;
        float frequency = 1f;
        float totalAmplitude = 0f;

        for (int i = 0; i < 3; i++) // Reduced octaves for better performance
        {
            noise += amplitude * GetNoise(
                position.X * NOISE_SCALE * frequency,
                position.Z * NOISE_SCALE * frequency,
                0 // We only need 2D noise for the height map
            );
            totalAmplitude += amplitude;
            amplitude *= 0.5f;
            frequency *= 2f;
        }

        return noise / totalAmplitude;
    }

    private bool IsCaveLocation(Vector3 position)
    {
        // Generate 3D noise for cave systems
        float caveNoise = GetNoise(position * CAVE_SCALE);

        // Additional noise for cave tunnels
        float tunnelNoise = GetNoise(position * CAVE_SCALE * 2);

        return caveNoise > CAVE_THRESHOLD || tunnelNoise > CAVE_THRESHOLD;
    }

    private (int, Vector4) GenerateCaveFeatures(Vector3 position)
    {
        // Generate stalagmites and stalactites
        float verticalNoise = GetFlatNoise(position * CAVE_SCALE * 2);

        if (verticalNoise > 0.7f)
        {
            // Stone formations
            return (1, palette[12]); // Gray stone color
        }

        // Cave air
        return (0, palette[1]); // Black for cave interior
    }

    private (int, Vector4) GetBiomeFeatures(Vector3 position, float heightRatio, float biomeNoise, float chunkWidth)
    {
        if (biomeNoise > MOUNTAIN_THRESHOLD)
        {
            if (heightRatio > 0.8f)
            {
                return (1, palette[0]); // Snow caps
            }
            return (1, palette[12]); // Mountain rock
        }
        else if (biomeNoise > PLAINS_THRESHOLD)
        {
            if (heightRatio > 0.6f)
            {
                float grassNoise = GetNoise(position.X * NOISE_SCALE * 4, position.Z * NOISE_SCALE * 4, 0);
                return grassNoise > 0.5f ? (1, palette[17]) : (1, palette[8]); // Mix of grass types
            }
            return (1, palette[5]); // Dirt/soil
        }
        else if (biomeNoise > DESERT_THRESHOLD)
        {
            return (1, palette[19]); // Desert sand
        }
        else // Forest
        {
            if (heightRatio > 0.55f)
            {
                float forestNoise = GetNoise(position.X * NOISE_SCALE * 3, position.Z * NOISE_SCALE * 3, 0);
                return forestNoise > 0.6f ? (1, palette[7]) : (1, palette[6]); // Mix of forest colors
            }
            return (1, palette[3]); // Forest floor
        }
    }
}