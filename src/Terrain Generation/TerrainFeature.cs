
using System.Numerics;

public interface ITerrainFeature
{
    bool ShouldGenerate(Vector3 worldPosition);
    void Generate(Vector3 localPosition, Vector3 worldPosition, Dictionary<Vector3, (int, Vector4)> blocks);
    Vector3 GetBounds(); // Returns the size of the feature in voxels
}