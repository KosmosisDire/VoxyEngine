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
}