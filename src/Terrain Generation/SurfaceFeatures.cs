using System.Numerics;


public abstract class SurfaceFeature : ITerrainFeature
{
    protected readonly float probability;

    protected SurfaceFeature(float probability = 1.0f)
    {
        this.probability = probability;
    }

    public virtual bool ShouldGenerate(Vector3 worldPosition)
    {
        if (StaticRandom.NextSingle() > probability) return false;
        return true;
    }

    public void Generate(Vector3 localPosition, Vector3 worldPosition, Dictionary<Vector3, (int, Vector4)> blocks)
    {
        if (!CanGenerateAt(localPosition, blocks)) return;
        GenerateFeature(localPosition, worldPosition, blocks);
    }

    protected virtual bool CanGenerateAt(Vector3 localPosition, Dictionary<Vector3, (int, Vector4)> blocks)
    {
        // Check if we have enough space in local coordinates
        var bounds = GetBounds();
        for (int y = 0; y < bounds.Y; y++)
        for (int x = (int)(-bounds.X/2.0); x <= bounds.X/2; x++)
        for (int z = (int)(-bounds.Z/2.0); z <= bounds.Z/2; z++)
        {
            Vector3 checkPos = localPosition + new Vector3(x, y, z);
            if (blocks.ContainsKey(checkPos)) return false;
        }
        return true;
    }

    public abstract Vector3 GetBounds();
    protected abstract void GenerateFeature(Vector3 localPosition, Vector3 worldPosition, Dictionary<Vector3, (int, Vector4)> blocks);
}

public class PineTree : SurfaceFeature
{
    private readonly int minHeight;
    private readonly int maxHeight;
    private readonly int leafLayers;
    private readonly float leafDensity;

    public PineTree(int minHeight = 6, int maxHeight = 12, int leafLayers = 6, 
        float leafDensity = 0.8f, float probability = 1.0f) : base(probability)
    {
        this.minHeight = minHeight;
        this.maxHeight = maxHeight;
        this.leafLayers = leafLayers;
        this.leafDensity = leafDensity;
    }

    public override Vector3 GetBounds()
    {
        int width = leafLayers * 2 + 1;
        return new Vector3(width, maxHeight, width);
    }

    protected override void GenerateFeature(Vector3 localPosition, Vector3 worldPosition, Dictionary<Vector3, (int, Vector4)> blocks)
    {
        int height = StaticRandom.Next(minHeight, maxHeight + 1);
        
        // Generate trunk
        for (int y = 0; y < height; y++)
        {
            Vector3 trunkPos = localPosition + new Vector3(0, y, 0);
            blocks[trunkPos] = (1, TerrainGenerator.Palette[3]); // Wood color
        }

        // Generate pine leaves in a cone shape
        float leafStart = height * 0.4f;
        for (int layer = 0; layer < leafLayers; layer++)
        {
            int layerY = height - (leafLayers - layer);
            if (layerY < leafStart) continue;

            int radius = Math.Max(1, (int)((leafLayers - layer) * 0.9f));

            for (int x = -radius; x <= radius; x++)
            for (int z = -radius; z <= radius; z++)
            {
                float distanceFromCenter = MathF.Sqrt(x * x + z * z);
                if (distanceFromCenter > radius) continue;

                if (StaticRandom.NextSingle() > leafDensity) continue;

                Vector3 leafPos = localPosition + new Vector3(x, layerY, z);
                if (!blocks.ContainsKey(leafPos))
                {
                    blocks[leafPos] = (1, TerrainGenerator.Palette[7]); // Leaf color
                }
            }
        }
    }
}

public class OakTree : SurfaceFeature
{
    private readonly int minHeight;
    private readonly int maxHeight;
    private readonly int minRadius;
    private readonly int maxRadius;
    private readonly float leafDensity;

    public OakTree(int minHeight = 5, int maxHeight = 8, int minRadius = 2, 
        int maxRadius = 4, float leafDensity = 0.7f, float probability = 1.0f) : base(probability)
    {
        this.minHeight = minHeight;
        this.maxHeight = maxHeight;
        this.minRadius = minRadius;
        this.maxRadius = maxRadius;
        this.leafDensity = leafDensity;
    }

    public override Vector3 GetBounds()
    {
        int width = maxRadius * 2 + 1;
        return new Vector3(width, maxHeight, width);
    }

    protected override void GenerateFeature(Vector3 localPosition, Vector3 worldPosition, Dictionary<Vector3, (int, Vector4)> blocks)
    {
        int height = StaticRandom.Next(minHeight, maxHeight + 1);
        
        // Generate trunk
        for (int y = 0; y < height; y++)
        {
            Vector3 trunkPos = localPosition + new Vector3(0, y, 0);
            blocks[trunkPos] = (1, TerrainGenerator.Palette[3]); // Wood color
        }

        // Generate leaf cluster positions
        int numClusters = StaticRandom.Next(3, 6);
        var clusters = new List<(Vector3 pos, int radius)>();
        
        // Main top cluster in local space
        clusters.Add((localPosition + new Vector3(0, height, 0), StaticRandom.Next(minRadius, maxRadius + 1)));

        // Additional clusters
        for (int i = 1; i < numClusters; i++)
        {
            float angle = StaticRandom.NextSingle() * MathF.PI * 2;
            float distance = StaticRandom.Next(1, 3);
            Vector3 offset = new Vector3(
                MathF.Cos(angle) * distance,
                -StaticRandom.Next(1, 3),
                MathF.Sin(angle) * distance
            );
            
            clusters.Add((clusters[0].pos + offset, StaticRandom.Next(minRadius - 1, maxRadius)));
        }

        foreach (var (clusterPos, radius) in clusters)
        {
            GenerateLeafCluster(clusterPos, radius, blocks);
        }
    }

    private void GenerateLeafCluster(Vector3 center, int radius, Dictionary<Vector3, (int, Vector4)> blocks)
    {
        for (int x = -radius; x <= radius; x++)
        for (int y = -radius; y <= radius; y++)
        for (int z = -radius; z <= radius; z++)
        {
            Vector3 offset = new Vector3(x, y, z);
            float distSq = offset.LengthSquared();
            if (distSq > radius * radius) continue;
            
            float edgeRatio = distSq / (radius * radius);
            if (StaticRandom.NextSingle() > leafDensity * (1 - edgeRatio * 0.5f)) continue;

            Vector3 leafPos = center + offset;
            if (!blocks.ContainsKey(leafPos))
            {
                blocks[leafPos] = (1, TerrainGenerator.Palette[6]); // Leaf color
            }
        }
    }
}

public class Rock : SurfaceFeature
{
    private readonly int minSize;
    private readonly int maxSize;
    private readonly float roughness;

    public Rock(int minSize = 1, int maxSize = 3, float roughness = 0.3f, 
        float probability = 1.0f) : base(probability)
    {
        this.minSize = minSize;
        this.maxSize = maxSize;
        this.roughness = roughness;
    }

    public override Vector3 GetBounds()
    {
        return new Vector3(maxSize * 2 + 1, maxSize + 1, maxSize * 2 + 1);
    }

    protected override void GenerateFeature(Vector3 localPosition, Vector3 worldPosition, Dictionary<Vector3, (int, Vector4)> blocks)
    {
        int size = StaticRandom.Next(minSize, maxSize + 1);
        
        for (int x = -size; x <= size; x++)
        for (int y = 0; y <= size; y++)
        for (int z = -size; z <= size; z++)
        {
            Vector3 offset = new Vector3(x, y, z);
            float distToCenter = offset.Length();
            if (distToCenter > size) continue;

            float noise = StaticRandom.NextSingle() * roughness;
            if (distToCenter > size * (1 - roughness) && StaticRandom.NextSingle() < noise)
                continue;

            Vector3 rockPos = localPosition + offset;
            if (!blocks.ContainsKey(rockPos))
            {
                int colorIndex = StaticRandom.NextSingle() < 0.3f ? 13 : 12;
                blocks[rockPos] = (1, TerrainGenerator.Palette[colorIndex]);
            }
        }
    }
}

public class Flowers : SurfaceFeature
{
    private readonly int maxRadius;
    private readonly float density;
    private readonly Vector4[] flowerColors;

    public Flowers(int maxRadius = 3, float density = 0.1f, 
        float probability = 1.0f) : base(probability)
    {
        this.maxRadius = maxRadius;
        this.density = density;
        
        this.flowerColors = new[]
        {
            new Vector4(1.0f, 0.4f, 0.4f, 1.0f), // Red
            new Vector4(1.0f, 1.0f, 0.4f, 1.0f), // Yellow
            new Vector4(0.4f, 0.4f, 1.0f, 1.0f), // Blue
            new Vector4(1.0f, 0.6f, 0.8f, 1.0f), // Pink
            new Vector4(1.0f, 0.8f, 0.4f, 1.0f)  // Orange
        };
    }

    public override Vector3 GetBounds()
    {
        return new Vector3(maxRadius * 2 + 1, 1, maxRadius * 2 + 1);
    }

    protected override void GenerateFeature(Vector3 localPosition, Vector3 worldPosition, Dictionary<Vector3, (int, Vector4)> blocks)
    {
        int radius = StaticRandom.Next(1, maxRadius + 1);
        Vector4 flowerColor = flowerColors[StaticRandom.Next(flowerColors.Length)];

        for (int x = -radius; x <= radius; x++)
        for (int z = -radius; z <= radius; z++)
        {
            if (StaticRandom.NextSingle() > density) continue;

            Vector3 flowerPos = localPosition + new Vector3(x, 0, z);
            if (!blocks.ContainsKey(flowerPos))
            {
                blocks[flowerPos] = (1, flowerColor);
            }
        }
    }
}