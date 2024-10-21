using System.Numerics;

namespace Engine;

public struct Bounds
{
    public Vector3 min;
    public Vector3 max;

    public Bounds(Vector3 min, Vector3 max)
    {
        this.min = min;
        this.max = max;
    }

    public bool Contains(Vector3 point)
    {
        return point.X >= min.X && point.X <= max.X &&
               point.Y >= min.Y && point.Y <= max.Y &&
               point.Z >= min.Z && point.Z <= max.Z;
    }

    public bool Intersects(Bounds bounds)
    {
        return min.X <= bounds.max.X && max.X >= bounds.min.X &&
               min.Y <= bounds.max.Y && max.Y >= bounds.min.Y &&
               min.Z <= bounds.max.Z && max.Z >= bounds.min.Z;
    }

    public static Bounds FromCenter(Vector3 center, Vector3 size)
    {
        return new Bounds(center - size / 2, center + size / 2);
    }

    public Bounds Transform(Matrix4x4 transform)
    {
        Vector3[] corners =
        [
            new Vector3(min.X, min.Y, min.Z),
            new Vector3(min.X, min.Y, max.Z),
            new Vector3(min.X, max.Y, min.Z),
            new Vector3(min.X, max.Y, max.Z),
            new Vector3(max.X, min.Y, min.Z),
            new Vector3(max.X, min.Y, max.Z),
            new Vector3(max.X, max.Y, min.Z),
            new Vector3(max.X, max.Y, max.Z),
        ];

        for (int i = 0; i < corners.Length; i++)
        {
            corners[i] = Vector3.Transform(corners[i], transform);
        }

        Vector3 newMin = new Vector3(float.MaxValue);
        Vector3 newMax = new Vector3(float.MinValue);

        foreach (var corner in corners)
        {
            newMin = Vector3.Min(newMin, corner);
            newMax = Vector3.Max(newMax, corner);
        }

        return new Bounds(newMin, newMax);
    }
}
