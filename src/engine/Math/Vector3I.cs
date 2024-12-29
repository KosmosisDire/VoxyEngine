using System.Numerics;

public struct Vector3I
{
    public int X, Y, Z;

    public Vector3I(int x, int y, int z)
    {
        X = x;
        Y = y;
        Z = z;
    }

    public Vector3I(int s)
    {
        X = s;
        Y = s;
        Z = s;
    }

    public Vector3I(uint s)
    {
        X = (int)s;
        Y = (int)s;
        Z = (int)s;
    }

    public Vector3I(Vector3 v)
    {
        X = (int)v.X;
        Y = (int)v.Y;
        Z = (int)v.Z;
    }

    public static Vector3I operator +(Vector3I a, Vector3I b)
    {
        return new Vector3I(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    }

    public static Vector3I operator -(Vector3I a, Vector3I b)
    {
        return new Vector3I(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
    }

    public static Vector3I operator *(Vector3I a, int b)
    {
        return new Vector3I(a.X * b, a.Y * b, a.Z * b);
    }

    public static Vector3I operator /(Vector3I a, int b)
    {
        return new Vector3I(a.X / b, a.Y / b, a.Z / b);
    }

    public static bool operator ==(Vector3I a, Vector3I b)
    {
        return a.X == b.X && a.Y == b.Y && a.Z == b.Z;
    }

    public static bool operator !=(Vector3I a, Vector3I b)
    {
        return a.X != b.X || a.Y != b.Y || a.Z != b.Z;
    }

    public override bool Equals(object obj)
    {
        return obj is Vector3I vector && this == vector;
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(X, Y, Z);
    }

    public override string ToString()
    {
        return $"({X}, {Y}, {Z})";
    }

    public static Vector3I Zero => new Vector3I(0, 0, 0);
    public static Vector3I One => new Vector3I(1, 1, 1);

    public static Vector3I UnitX => new Vector3I(1, 0, 0);
    public static Vector3I UnitY => new Vector3I(0, 1, 0);
    public static Vector3I UnitZ => new Vector3I(0, 0, 1);

    public static implicit operator Vector3(Vector3I v)
    {
        return new Vector3(v.X, v.Y, v.Z);
    }

    public static implicit operator Vector3I(Vector3 v)
    {
        return new Vector3I((int)v.X, (int)v.Y, (int)v.Z);
    }
}

