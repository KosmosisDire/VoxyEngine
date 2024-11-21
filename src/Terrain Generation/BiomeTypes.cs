
using System.Numerics;

public class MountainBiome : BiomeBase
{
    public MountainBiome() : base(5) { }

    public override (int, Vector4) GetBlockType(Vector3 position, float heightRatio)
    {
        if (heightRatio > 0.8f)
            return (1, TerrainGenerator.Palette[0]); // Snow
        if (heightRatio > 0.6f)
            return (1, TerrainGenerator.Palette[12]); // Stone
        return (1, TerrainGenerator.Palette[14]); // Dark stone
    }

    public override IEnumerable<SurfaceFeature> GetFeatures(Vector3 position)
    {
        yield break; // Mountains don't have additional features
    }
}

public class ForestBiome : BiomeBase
{
    private readonly List<ITerrainFeature> features;
    private readonly float treeNoiseScale = 0.05f;
    private readonly float treeSpacing = 0.2f;

    public ForestBiome() : base(2)
    {
        features = new List<ITerrainFeature>
        {

            new OakTree(probability: 0.3f),
            new PineTree(probability: 0.3f),
            new Rock(probability: 0.1f),
            new Flowers(probability: 0.2f)
        };
    }

    public override (int, Vector4) GetBlockType(Vector3 position, float heightRatio)
    {
        if (heightRatio > 0.55f)
        {
            float forestNoise = TerrainGenerator.GetNoise(position * TerrainGenerator.NOISE_SCALE * 3);
            return forestNoise > 0.6f ? 
                (1, TerrainGenerator.Palette[7]) : // Dark forest
                (1, TerrainGenerator.Palette[6]);  // Light forest
        }
        return (1, TerrainGenerator.Palette[3]); // Forest floor
    }

    public override IEnumerable<ITerrainFeature> GetFeatures(Vector3 position)
    {
        return features;
    }
}
