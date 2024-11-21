using System.Numerics;

// Update biome interface to use ITerrainFeature
public interface IBiome
{
    float GetSurfaceHeight(Vector3 position, float baseHeight);
    float GetPriority(Vector3 position);
    (int, Vector4) GetBlockType(Vector3 position, float heightRatio);
    IEnumerable<ITerrainFeature> GetFeatures(Vector3 position);
}


public abstract class BiomeBase : IBiome
{
    protected readonly float heightScale;
    protected readonly float priorityNoiseScale;
    
    protected BiomeBase(float heightScale = 1.0f, float priorityNoiseScale = 0.01f)
    {
        this.heightScale = heightScale;
        this.priorityNoiseScale = priorityNoiseScale;
    }

    public virtual float GetSurfaceHeight(Vector3 position, float baseHeight)
    {
        return baseHeight * heightScale;
    }

    public virtual float GetPriority(Vector3 position)
    {
        return TerrainGenerator.GetFlatNoise(position * priorityNoiseScale);
    }

    public abstract (int, Vector4) GetBlockType(Vector3 position, float heightRatio);
    public abstract IEnumerable<ITerrainFeature> GetFeatures(Vector3 position);
}
