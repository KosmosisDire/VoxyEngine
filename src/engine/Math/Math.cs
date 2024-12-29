using System.Numerics;

namespace Engine;


public static class EngineMath
{
    private static readonly Random Random = new();

    public static float Lerp(float a, float b, float t)
    {
        return a + (b - a) * t;
    }

    public static float Clamp01(float value)
    {
        return value < 0 ? 0 : value > 1 ? 1 : value;
    }

    public static Vector3 GetRandomDirection()
    {
        return new Vector3(
            (float)Random.NextDouble() * 2 - 1,
            (float)Random.NextDouble() * 2 - 1,
            (float)Random.NextDouble() * 2 - 1
        );
    }

    public static Vector3 GetRandomDirection(Vector3 normal)
    {
        var direction = GetRandomDirection();
        while (Vector3.Dot(direction, normal) < 0)
        {
            direction = GetRandomDirection();
        }
        return direction;
    }

    public static float SimplexNoisePixel(int x, int y, int z, float scale = 1)
    {
        return Simplex.CalcPixel3D(x, y, z, scale);
    }

    public static float SimplexNoisePixel(Vector3I pos, float scale = 1)
    {
        return Simplex.CalcPixel3D(pos.X, pos.Y, pos.Z, scale);
    }

    public static float SimplexNoisePixel(int x, int y, float scale = 1)
    {
        return Simplex.CalcPixel2D(x, y, scale);
    }

    public static float SimplexNoisePixel(Vector2I pos, float scale = 1)
    {
        return Simplex.CalcPixel2D(pos.X, pos.Y, scale);
    }

    public static float SimplexNoisePixel(int x, float scale = 1)
    {
        return Simplex.CalcPixel1D(x, scale);
    }

    public static float SimplexNoise3D(float x, float y, float z, float frequency = 10f, float scale = 1f)
    {
        int ix = (int)MathF.Floor(x * frequency);
        int iy = (int)MathF.Floor(y * frequency);
        int iz = (int)MathF.Floor(z * frequency);

        return SimplexNoisePixel(ix, iy, iz, scale);
    }

    public static float SimplexNoise2D(float x, float y, float frequency = 10f, float scale = 1f)
    {
        int ix = (int)MathF.Floor(x * frequency);
        int iy = (int)MathF.Floor(y * frequency);

        return SimplexNoisePixel(ix, iy, scale);
    }

    public static float SimplexNoise1D(float x, float frequency = 10f, float scale = 1f)
    {
        int ix = (int)MathF.Floor(x * frequency);

        return SimplexNoisePixel(ix, scale);
    }
}